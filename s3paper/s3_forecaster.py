"""Canonical causal S3-Forecaster implementation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor

from .calibration import split_internal_calibration
from .conformal import SequentialACI, finalize_symmetric_interval
from .priors import (
    CausalPrior,
    SimpleFoundationProxy,
    build_prior,
    ensure_causal_prior,
    normalize_prior_name,
)
from .preprocessing import FrozenRobustScaler
from .residual_features import EchoStateResidualTransformer
from .selection import AdapterSelectionRule, residual_predictability_score
from .utils import ensure_series


EchoStateFeatureExtractor = EchoStateResidualTransformer


class S3Forecaster:
    """Small-data residual adapter with a deterministic ESN representation."""

    def __init__(
        self,
        reservoir_size: int = 50,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.5,
        ar_lags: int = 3,
        horizon: int = 1,
        aci_step_size: float = 0.05,
        foundation_window: int = 6,
        prior_name: str = "causal_rolling_mean",
        prior_params: dict[str, Any] | None = None,
        regressor_type: str = "Ridge",
        reg_alpha: float = 1.0,
        calibration_split_ratio: float = 0.7,
        oob_split_ratio: float | None = None,
        target_miscoverage: float = 0.10,
        seed: int = 42,
        foundation: CausalPrior | None = None,
        use_residual_adapter: bool = True,
        use_volatility_gate: bool = True,
        use_aci: bool = True,
        use_adapter_selector: bool = True,
        disabled_groups: set[str] | None = None,
        selection_r2_threshold: float = -np.inf,
        selection_minimum_improvement: float = -np.inf,
        selection_complexity_penalty: float = 0.0,
        interval_scale: float = 1.0,
        minimum_width: float = 0.0,
        prior_cache: Any = None,
        prior_cache_context: dict[str, Any] | None = None,
    ):
        if oob_split_ratio is not None:
            warnings.warn(
                "oob_split_ratio is deprecated; use calibration_split_ratio.",
                DeprecationWarning,
                stacklevel=2,
            )
            calibration_split_ratio = float(oob_split_ratio)

        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(min(max(spectral_radius, 1e-6), 0.999))
        self.leak_rate = float(leak_rate)
        self.ar_lags = int(ar_lags)
        self.horizon = int(horizon)
        self.aci_step_size = float(aci_step_size)
        self.foundation_window = int(foundation_window)
        self.prior_name = normalize_prior_name(str(prior_name))
        self.prior_params = dict(prior_params or {})
        self.regressor_type = str(regressor_type)
        self.reg_alpha = float(reg_alpha)
        self.calibration_split_ratio = float(calibration_split_ratio)
        self.target_miscoverage = float(target_miscoverage)
        self.seed = int(seed)
        self.foundation = foundation
        self.use_residual_adapter = bool(use_residual_adapter)
        self.use_volatility_gate = bool(use_volatility_gate)
        self.use_aci = bool(use_aci)
        self.use_adapter_selector = bool(use_adapter_selector)
        self.disabled_groups = set(disabled_groups or set())
        self.selection_rule = AdapterSelectionRule(
            r2_threshold=selection_r2_threshold,
            minimum_improvement=selection_minimum_improvement,
            complexity_penalty=selection_complexity_penalty,
        )
        self.interval_scale = float(interval_scale)
        self.minimum_width = float(minimum_width)
        if self.interval_scale <= 0.0:
            raise ValueError("interval_scale must be > 0.")
        if self.minimum_width < 0.0:
            raise ValueError("minimum_width must be >= 0.")
        self.prior_cache = prior_cache
        self.prior_cache_context = prior_cache_context

        self.transformer = EchoStateResidualTransformer(
            reservoir_size=self.reservoir_size,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            ar_lags=self.ar_lags,
            seed=self.seed,
        )
        self.volatility_scaler_ = FrozenRobustScaler()
        self.readout = self._build_readout()
        self.aci_ = SequentialACI(
            target_miscoverage=self.target_miscoverage,
            step_size=self.aci_step_size,
        )
        self.fitted = False
        self.fit_report_: dict[str, Any] = {}
        self.selector_report_: dict[str, Any] = {}

    def _make_prior(self) -> CausalPrior:
        if self.foundation is not None:
            prior = self.foundation.clone() if hasattr(self.foundation, "clone") else self.foundation
            return ensure_causal_prior(prior)
        params = dict(self.prior_params)
        if self.prior_name == "causal_rolling_mean" and "window_size" not in params:
            params["window_size"] = self.foundation_window
        return build_prior(self.prior_name, **params)

    def _build_readout(self) -> MultiOutputRegressor:
        if self.regressor_type == "Ridge":
            base = Ridge(alpha=self.reg_alpha)
        elif self.regressor_type == "BayesianRidge":
            base = BayesianRidge()
        elif self.regressor_type == "Lasso":
            base = Lasso(alpha=self.reg_alpha, max_iter=10000)
        elif self.regressor_type == "ElasticNet":
            base = ElasticNet(alpha=self.reg_alpha, l1_ratio=0.5, max_iter=10000)
        else:
            raise ValueError(f"Unknown regressor_type={self.regressor_type!r}.")
        return MultiOutputRegressor(base)

    @staticmethod
    def _volatility(residuals: Any) -> np.ndarray:
        return (
            pd.Series(np.asarray(residuals, dtype=float).reshape(-1))
            .ewm(span=4, adjust=False)
            .std()
            .fillna(0.0)
            .to_numpy()
        )

    def _volatility_z(self, residuals: Any) -> np.ndarray:
        return self.volatility_scaler_.transform(self._volatility(residuals)).reshape(-1)

    def _gate(self, z_value: float) -> float:
        logits = np.clip(self.gate_alpha_ * (float(z_value) - self.gate_beta_), -50.0, 50.0)
        return float(1.0 / (1.0 + np.exp(logits)))

    def create_features(self, residuals: Any) -> np.ndarray:
        x_all = self.transformer.transform(residuals)
        names = list(getattr(self.transformer, "feature_names_", []))
        groups = list(getattr(self.transformer, "feature_groups_", []))
        if not self.disabled_groups:
            self.feature_names_ = names
            self.feature_groups_ = groups
            return x_all
        keep = np.asarray([group not in self.disabled_groups for group in groups], dtype=bool)
        if len(keep) != x_all.shape[1] or keep.sum() == 0:
            self.feature_names_ = ["constant_zero"]
            self.feature_groups_ = ["constant"]
            return np.zeros((len(x_all), 1), dtype=float)
        self.feature_names_ = [name for name, flag in zip(names, keep) if flag]
        self.feature_groups_ = [group for group, flag in zip(groups, keep) if flag]
        return x_all[:, keep]

    def _calibration_predictions(
        self,
        train_block: pd.Series,
        calibration_block: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.foundation is not None:
            prior = self.foundation.clone() if hasattr(self.foundation, "clone") else self.foundation
            from .priors import ensure_causal_prior
            prior = ensure_causal_prior(prior).fit(train_block)
            base_train = prior.fitted_values().to_numpy(dtype=float)
            base_cal = []
            for obs in calibration_block.to_numpy(dtype=float):
                base_cal.append(float(prior.predict(1).iloc[0]))
                prior.update(float(obs))
            base_cal = np.asarray(base_cal, dtype=float)
        else:
            from .prior_cache import compute_causal_prior_path
            base_train, base_cal = compute_causal_prior_path(
                self.prior_name,
                self.prior_params,
                train_block,
                calibration_block,
                cache=self.prior_cache,
                cache_context=self.prior_cache_context,
            )

        residual_history = (
            train_block.to_numpy(dtype=float)
            - base_train
        ).tolist()
        base_values: list[float] = []
        raw_values: list[float] = []
        target_residuals: list[float] = []
        z_values: list[float] = []

        for i, obs in enumerate(calibration_block.to_numpy(dtype=float)):
            base = float(base_cal[i])
            x_last = self.create_features(residual_history)[-1].reshape(1, -1)
            raw = float(self.readout.predict(x_last)[0, 0])
            z = float(self._volatility_z(residual_history)[-1])
            target_residual = float(obs - base)
            base_values.append(base)
            raw_values.append(raw)
            target_residuals.append(target_residual)
            z_values.append(z)
            residual_history.append(target_residual)

        return (
            np.asarray(base_values, dtype=float),
            np.asarray(raw_values, dtype=float),
            np.asarray(target_residuals, dtype=float),
            np.asarray(z_values, dtype=float),
        )

    def fit(self, series: Any):
        y = ensure_series(series)
        n = len(y)
        self.fit_report_ = {
            "n": int(n),
            "horizon": int(self.horizon),
            "status": "initializing",
            "prior_name": self.prior_name,
        }
        self.fitted = False
        try:
            split = split_internal_calibration(
                y,
                readout_ratio=self.calibration_split_ratio,
                adapter_ratio=0.20,
                minimum_readout=self.ar_lags + 8,
                minimum_adapter=3,
                minimum_conformal=3,
            )
        except ValueError as exc:
            self.fit_report_["status"] = str(exc)
            return self

        readout_block = split.readout_train
        adapter_block = split.adapter_calibration
        conformal_block = split.conformal_calibration
        train_prior = self._make_prior().fit(readout_block)
        train_residual = readout_block.to_numpy(dtype=float) - train_prior.fitted_values().to_numpy(dtype=float)

        self.transformer.fit(train_residual)
        x_train = self.create_features(train_residual)
        y_train = np.roll(train_residual, -1)
        valid = np.arange(len(train_residual) - 1)
        valid = valid[valid >= self.ar_lags]
        if len(valid) < 2:
            self.fit_report_["status"] = "insufficient_readout_samples"
            return self

        self.readout = self._build_readout()
        self.readout.fit(x_train[valid], y_train[valid].reshape(-1, 1))
        self.volatility_scaler_.fit(self._volatility(train_residual))

        base_cal, raw_cal, target_resid_cal, z_cal = self._calibration_predictions(
            readout_block,
            adapter_block,
        )

        def gate_loss(params: np.ndarray) -> float:
            alpha, beta = params
            logits = np.clip(float(alpha) * (z_cal - float(beta)), -50.0, 50.0)
            gate = 1.0 / (1.0 + np.exp(logits))
            return float(np.mean(np.abs(target_resid_cal - raw_cal * gate)))

        optimum = minimize(gate_loss, [1.0, 0.0], bounds=[(0.01, 20.0), (-5.0, 5.0)])
        self.gate_alpha_, self.gate_beta_ = map(float, optimum.x)
        if self.use_volatility_gate:
            gate = 1.0 / (1.0 + np.exp(np.clip(self.gate_alpha_ * (z_cal - self.gate_beta_), -50.0, 50.0)))
        else:
            gate = np.ones_like(raw_cal)
        adapted = base_cal + raw_cal * gate
        prior_loss = float(np.mean(np.abs(adapter_block.to_numpy(dtype=float) - base_cal)))
        adapted_loss = float(np.mean(np.abs(adapter_block.to_numpy(dtype=float) - adapted)))
        r2_res_cal = residual_predictability_score(target_resid_cal, raw_cal)
        self.selector_report_ = self.selection_rule.decide(
            r2_res_cal=r2_res_cal,
            delta_cal=prior_loss - adapted_loss,
            prior_loss=prior_loss,
            adapted_loss=adapted_loss,
            reason_prefix="s3_adapter_calibration",
        )
        if not self.use_residual_adapter:
            self.selector_report_["activated"] = False
            self.selector_report_["reason"] = "adapter_disabled_by_ablation"
        elif not self.use_adapter_selector:
            self.selector_report_["activated"] = True
            self.selector_report_["reason"] = "adapter_selector_disabled_by_ablation"

        base_conf, raw_conf, _, z_conf = self._calibration_predictions(
            pd.concat([readout_block, adapter_block]),
            conformal_block,
        )
        if self.use_volatility_gate:
            gate_conf = 1.0 / (1.0 + np.exp(np.clip(self.gate_alpha_ * (z_conf - self.gate_beta_), -50.0, 50.0)))
        else:
            gate_conf = np.ones_like(raw_conf)
        correction_conf = raw_conf * gate_conf if self.selector_report_.get("activated", False) else np.zeros_like(raw_conf)
        adapted_conf = base_conf + correction_conf
        self.aci_ = SequentialACI(self.target_miscoverage, self.aci_step_size).fit_calibration(
            conformal_block.to_numpy(dtype=float),
            adapted_conf,
        )

        self.prior_ = self._make_prior().fit(y)
        self.last_series = y.copy()
        self.last_prediction_: dict[str, float] | None = None
        self.fit_report_.update(
            {
                "status": "success",
                "n_readout_samples": int(len(valid)),
                "n_calibration_samples": int(len(adapter_block) + len(conformal_block)),
                "n_features": int(x_train.shape[1]),
                "achieved_spectral_radius": self.transformer.achieved_spectral_radius_,
                **split.report(),
            }
        )
        self.fitted = True
        return self

    def _residual_history(self) -> np.ndarray:
        return self.last_series.to_numpy(dtype=float) - self.prior_.fitted_values().to_numpy(dtype=float)

    def predict_one(self) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(f"S3Forecaster is not fitted: {self.fit_report_}.")
        base_forecast = self.prior_.predict(1)
        base = float(base_forecast.iloc[0])
        residuals = self._residual_history()
        raw = float(self.readout.predict(self.create_features(residuals)[-1].reshape(1, -1))[0, 0])
        gate = self._gate(float(self._volatility_z(residuals)[-1])) if self.use_volatility_gate else 1.0
        residual_correction = gate * raw if self.selector_report_.get("activated", False) else 0.0
        pred = float(base + residual_correction)
        raw_interval = self.aci_.interval(pred) if self.use_aci else (pred, pred)
        lower, upper, half_width = finalize_symmetric_interval(
            pred,
            raw_interval,
            interval_scale=self.interval_scale,
            minimum_width=self.minimum_width,
        )
        self.last_prediction_ = {
            "base": base,
            "pred": pred,
            "raw": raw,
            "gate": gate,
            "lower": lower,
            "upper": upper,
            "half_width": half_width,
        }
        return pd.DataFrame(
            {
                "base": [base],
                "pred": [pred],
                "lower": [lower],
                "upper": [upper],
                "gate": [gate],
                "residual_raw": [raw],
                "residual_corrected": [residual_correction],
                "half_width": [half_width],
            },
            index=base_forecast.index,
        )

    def update(self, observation: float, *, is_observed: bool = True):
        if is_observed and self.last_prediction_ is not None:
            self.aci_.update(
                float(observation),
                float(self.last_prediction_["pred"]),
                interval=(float(self.last_prediction_["lower"]), float(self.last_prediction_["upper"])),
            )
        self.prior_.update(float(observation))
        next_index = self.prior_.fitted_values().index[-1]
        self.last_series.loc[next_index] = float(observation)
        self.last_prediction_ = None
        return self

    def predict(self, steps: int | None = None, recursive: bool = True) -> pd.DataFrame:
        steps = int(steps or self.horizon)
        rows = []
        for _ in range(steps):
            row = self.predict_one()
            rows.append(row)
            self.update(float(row["pred"].iloc[0]), is_observed=False)
        return pd.concat(rows)

    def get_point_components(self) -> dict[str, float]:
        return dict(self.last_prediction_ or {})

    @property
    def conformity_scores(self) -> np.ndarray:
        return np.asarray(self.aci_.scores, dtype=float)

    @property
    def current_alpha_t(self) -> float:
        return float(self.aci_.alpha_t)

    def trainable_parameter_count(self) -> int:
        total = 0
        for estimator in getattr(self.readout, "estimators_", []):
            total += np.asarray(getattr(estimator, "coef_", [])).size
            total += np.asarray(getattr(estimator, "intercept_", [])).size
        return int(total + 2)

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "fit_report": dict(self.fit_report_),
            "selector_report": dict(self.selector_report_),
            "prior_name": self.prior_name,
            "prior_params": dict(self.prior_params),
            "disabled_groups": sorted(self.disabled_groups),
            "use_adapter_selector": self.use_adapter_selector,
            "reservoir_size": self.reservoir_size,
            "achieved_spectral_radius": getattr(self.transformer, "achieved_spectral_radius_", np.nan),
            "leak_rate": self.leak_rate,
            "feature_dim": self.fit_report_.get("n_features"),
            "readout_samples": self.fit_report_.get("n_readout_samples"),
            "calibration_samples": self.fit_report_.get("n_calibration_samples"),
            "seed": self.seed,
            "gate_alpha": getattr(self, "gate_alpha_", np.nan),
            "gate_beta": getattr(self, "gate_beta_", np.nan),
            "current_alpha_t": self.current_alpha_t,
            "n_conformity_scores": len(self.conformity_scores),
            "trainable_params": self.trainable_parameter_count(),
        }


S3ForecasterV6 = S3Forecaster
