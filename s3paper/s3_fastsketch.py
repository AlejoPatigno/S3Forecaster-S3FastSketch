"""S3-FastSketch causal residual sketch forecaster."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .calibration import split_internal_calibration
from .conformal import SequentialACI, finalize_symmetric_interval
from .priors import SimpleFoundationProxy, build_prior, ensure_causal_prior, normalize_prior_name
from .residual_features import FastSketchResidualTransformer
from .selection import AdapterSelectionRule, residual_predictability_score
from .utils import ensure_series, make_future_index


def _make_forecast_index(series: pd.Series, horizon: int) -> pd.Index:
    return make_future_index(ensure_series(series), horizon)


class FastRollingFoundation(SimpleFoundationProxy):
    """Backward-compatible name for the causal rolling prior."""


class S3FastSketchForecaster:
    def __init__(
        self,
        horizon: int = 1,
        foundation_window: int = 6,
        prior_name: str = "causal_rolling_mean",
        prior_params: dict[str, Any] | None = None,
        ar_lags: int = 3,
        ema_spans: tuple[int, ...] = (2, 4, 8, 12),
        conv_scales: tuple[int, ...] = (2, 3, 4, 6),
        seasonal_period: int = 12,
        use_calendar: bool = True,
        ridge_alpha: float = 1.0,
        calibration_split_ratio: float = 0.64,
        oob_split_ratio: float | None = None,
        shrinkage_max: float = 1.5,
        signed_shrinkage: bool = False,
        aci_target: float = 0.10,
        aci_step_size: float = 0.05,
        min_train_samples: int = 8,
        min_calib_samples: int = 2,
        use_existing_simple_foundation: bool = True,
        foundation: Any | None = None,
        disabled_groups: set[str] | None = None,
        use_foundation: bool = True,
        foundation_only: bool = False,
        use_shrinkage: bool = True,
        use_aci: bool = True,
        use_adapter_selector: bool = True,
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
        self.horizon = int(horizon)
        self.foundation_window = int(foundation_window)
        self.prior_name = normalize_prior_name(str(prior_name))
        self.prior_params = dict(prior_params or {})
        self.ar_lags = int(ar_lags)
        self.ema_spans = tuple(ema_spans)
        self.conv_scales = tuple(conv_scales)
        self.seasonal_period = int(seasonal_period)
        self.use_calendar = bool(use_calendar)
        self.ridge_alpha = float(ridge_alpha)
        self.calibration_split_ratio = float(calibration_split_ratio)
        self.shrinkage_max = float(shrinkage_max)
        self.signed_shrinkage = bool(signed_shrinkage)
        self.aci_target = float(aci_target)
        self.aci_step_size = float(aci_step_size)
        self.min_train_samples = int(min_train_samples)
        self.min_calib_samples = int(min_calib_samples)
        self.use_existing_simple_foundation = bool(use_existing_simple_foundation)
        self.foundation = foundation
        self.disabled_groups = set(disabled_groups or set())
        self.use_foundation_component = bool(use_foundation)
        self.foundation_only = bool(foundation_only)
        self.use_shrinkage = bool(use_shrinkage)
        self.use_aci = bool(use_aci)
        self.use_adapter_selector = bool(use_adapter_selector)
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

        self.transformer = FastSketchResidualTransformer(
            ar_lags=self.ar_lags,
            ema_spans=self.ema_spans,
            conv_scales=self.conv_scales,
            seasonal_period=self.seasonal_period,
            use_calendar=self.use_calendar,
        )
        self.readout = Ridge(alpha=self.ridge_alpha)
        self.aci_ = SequentialACI(self.aci_target, self.aci_step_size)
        self.fitted = False
        self.fit_report_: dict[str, Any] = {}
        self.selector_report_: dict[str, Any] = {}
        self.gamma_: np.ndarray | None = None

    def _make_prior(self):
        if not self.use_foundation_component:
            return build_prior("causal_rolling_mean", window_size=10**9)
        if self.foundation is not None:
            prior = self.foundation.clone() if hasattr(self.foundation, "clone") else self.foundation
            return ensure_causal_prior(prior)
        params = dict(self.prior_params)
        if self.prior_name == "causal_rolling_mean" and "window_size" not in params:
            params["window_size"] = self.foundation_window
        return build_prior(self.prior_name, **params)

    @staticmethod
    def _clean_series(series: Any) -> pd.Series:
        return ensure_series(series)

    @staticmethod
    def _safe_array(x: Any) -> np.ndarray:
        return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def create_features(self, residuals: Any) -> np.ndarray:
        x_all = self.transformer.transform(residuals)
        names = list(self.transformer.feature_names_)
        groups = list(self.transformer.feature_groups_)
        if not self.disabled_groups:
            self.feature_names_ = names
            self.feature_groups_ = groups
            return x_all
        keep = np.asarray([group not in self.disabled_groups for group in groups], dtype=bool)
        if keep.sum() == 0:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_foundation_component:
            prior = self._make_prior().fit(train_block)
            base_train = prior.fitted_values().to_numpy(dtype=float)
            base_cal = []
            for obs in calibration_block.to_numpy(dtype=float):
                base_cal.append(float(prior.predict(1).iloc[0]))
                prior.update(float(obs))
            base_cal = np.asarray(base_cal, dtype=float)
        elif self.foundation is not None:
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
        for i, obs in enumerate(calibration_block.to_numpy(dtype=float)):
            base = float(base_cal[i])
            x_last = self.create_features(residual_history)[-1].reshape(1, -1)
            raw = float(self.readout.predict(x_last)[0])
            target_residual = float(obs - base)
            base_values.append(base)
            raw_values.append(raw)
            target_residuals.append(target_residual)
            residual_history.append(target_residual)
        return (
            np.asarray(base_values, dtype=float),
            np.asarray(raw_values, dtype=float),
            np.asarray(target_residuals, dtype=float),
        )

    def _estimate_gamma(self, raw_pred: np.ndarray, target_residual: np.ndarray) -> float:
        numerator = float(np.sum(raw_pred * target_residual))
        denominator = float(np.sum(raw_pred**2) + 1e-12)
        gamma = numerator / denominator
        lower = -self.shrinkage_max if self.signed_shrinkage else 0.0
        return float(np.clip(gamma, lower, self.shrinkage_max))

    def fit(self, series: Any):
        series = self._clean_series(series)
        n = len(series)
        self.fitted = False
        self.gamma_ = None
        self.fit_report_ = {
            "n": int(n),
            "horizon": int(self.horizon),
            "reason": "initializing",
            "prior_name": self.prior_name,
        }

        try:
            split = split_internal_calibration(
                series,
                readout_ratio=self.calibration_split_ratio,
                adapter_ratio=0.20,
                minimum_readout=self.ar_lags + self.min_train_samples,
                minimum_adapter=self.min_calib_samples,
                minimum_conformal=self.min_calib_samples,
            )
        except ValueError as exc:
            self.fit_report_["reason"] = str(exc)
            return self

        train_block = split.readout_train
        adapter_block = split.adapter_calibration
        conformal_block = split.conformal_calibration
        prior = self._make_prior().fit(train_block)
        residual_train = train_block.to_numpy(dtype=float) - prior.fitted_values().to_numpy(dtype=float)
        self.transformer.fit(residual_train)
        x_train = self.create_features(residual_train)
        y_train = np.roll(residual_train, -1)
        valid = np.arange(len(residual_train) - 1)
        valid = valid[valid >= self.ar_lags]
        if len(valid) < self.min_train_samples:
            self.fit_report_["reason"] = f"not_enough_readout_samples: {len(valid)}"
            return self

        self.readout = Ridge(alpha=self.ridge_alpha)
        self.readout.fit(x_train[valid], y_train[valid])

        base_cal, raw_cal, target_resid_cal = self._calibration_predictions(train_block, adapter_block)
        if len(raw_cal) < self.min_calib_samples:
            self.fit_report_["reason"] = f"calibration_block_too_small: {len(raw_cal)}"
            return self
        gamma = self._estimate_gamma(raw_cal, target_resid_cal)
        if self.foundation_only:
            gamma = 0.0
        elif not self.use_shrinkage:
            gamma = 1.0
        self.gamma_ = np.asarray([gamma], dtype=float)
        adapted = base_cal + gamma * raw_cal

        prior_loss = float(np.mean(np.abs(adapter_block.to_numpy(dtype=float) - base_cal)))
        adapted_loss = float(np.mean(np.abs(adapter_block.to_numpy(dtype=float) - adapted)))
        r2_res_cal = residual_predictability_score(target_resid_cal, raw_cal)
        self.selector_report_ = self.selection_rule.decide(
            r2_res_cal=r2_res_cal,
            delta_cal=prior_loss - adapted_loss,
            prior_loss=prior_loss,
            adapted_loss=adapted_loss,
            reason_prefix="fastsketch_calibration",
        )
        self.selector_report_.update(
            {
                "gamma": gamma,
                "signed_shrinkage": self.signed_shrinkage,
            }
        )
        if self.foundation_only:
            self.selector_report_["activated"] = False
            self.selector_report_["reason"] = "foundation_only_ablation"
        elif not self.use_shrinkage:
            self.selector_report_["activated"] = True
            self.selector_report_["reason"] = "shrinkage_disabled_ablation"
        elif not self.use_adapter_selector:
            self.selector_report_["activated"] = True
            self.selector_report_["reason"] = "adapter_selector_disabled_by_ablation"

        base_conf, raw_conf, _ = self._calibration_predictions(
            pd.concat([train_block, adapter_block]),
            conformal_block,
        )
        correction_conf = gamma * raw_conf if self.selector_report_.get("activated", False) else np.zeros_like(raw_conf)
        adapted_conf = base_conf + correction_conf
        self.aci_ = SequentialACI(self.aci_target, self.aci_step_size).fit_calibration(
            conformal_block.to_numpy(dtype=float),
            adapted_conf,
        )

        self.prior_ = self._make_prior().fit(series)
        self.last_series = series.copy()
        self.last_prediction_: dict[str, float] | None = None
        self.fit_report_.update(
            {
                "reason": "success",
                "n_features": int(x_train.shape[1]),
                "n_train_readout": int(len(valid)),
                "n_calib": int(len(adapter_block) + len(conformal_block)),
                **split.report(),
            }
        )
        self.fitted = True
        return self

    def _residual_history(self) -> np.ndarray:
        return self.last_series.to_numpy(dtype=float) - self.prior_.fitted_values().to_numpy(dtype=float)

    def predict_one(self) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(f"Model fit failed. Reason: {self.fit_report_}")
        base_forecast = self.prior_.predict(1)
        base = float(base_forecast.iloc[0])
        residuals = self._residual_history()
        raw = float(self.readout.predict(self.create_features(residuals)[-1].reshape(1, -1))[0])
        gamma = float(self.gamma_[0]) if self.gamma_ is not None else 0.0
        corrected = gamma * raw if self.selector_report_.get("activated", False) else 0.0
        pred = float(base + corrected)
        raw_interval = self.aci_.interval(pred) if self.use_aci else (pred, pred)
        lower, upper, half_width = finalize_symmetric_interval(
            pred,
            raw_interval,
            interval_scale=self.interval_scale,
            minimum_width=self.minimum_width,
        )
        self.last_prediction_ = {"base": base, "pred": pred, "raw": raw, "gamma": gamma, "lower": lower, "upper": upper, "half_width": half_width}
        return pd.DataFrame(
            {
                "step": [1],
                "base": [base],
                "pred": [pred],
                "lower": [lower],
                "upper": [upper],
                "residual_raw": [raw],
                "residual_corrected": [corrected],
                "gamma": [gamma],
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

    def predict(
        self,
        steps: int | None = None,
        recursive: bool = True,
    ) -> pd.DataFrame:
        del recursive
        steps = int(steps or self.horizon)
        rows = []
        for _ in range(steps):
            row = self.predict_one()
            rows.append(row)
            self.update(float(row["pred"].iloc[0]), is_observed=False)
        return pd.concat(rows)

    def feature_importance(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        coef = np.asarray(getattr(self.readout, "coef_", []), dtype=float).reshape(-1)
        return pd.DataFrame(
            {
                "horizon": 1,
                "feature": self.feature_names_,
                "group": self.feature_groups_,
                "coef": coef,
                "abs_coef": np.abs(coef),
            }
        ).sort_values("abs_coef", ascending=False)

    def group_importance(self) -> pd.DataFrame:
        fi = self.feature_importance()
        if fi.empty:
            return pd.DataFrame()
        return fi.groupby(["horizon", "group"], as_index=False)["abs_coef"].sum().sort_values("abs_coef", ascending=False)

    @property
    def conformity_scores(self) -> np.ndarray:
        return np.asarray(self.aci_.scores, dtype=float)

    @property
    def current_alpha_t(self) -> float:
        return float(self.aci_.alpha_t)

    def trainable_parameter_count(self) -> int:
        return int(np.asarray(getattr(self.readout, "coef_", [])).size + np.asarray(getattr(self.readout, "intercept_", [])).size + 1)

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "fit_report": self.fit_report_,
            "selector_report": self.selector_report_,
            "gamma": self.gamma_,
            "disabled_groups": sorted(self.disabled_groups),
            "use_adapter_selector": self.use_adapter_selector,
            "current_alpha_t": self.current_alpha_t,
            "n_conformity_scores": len(self.conformity_scores),
            "n_features": self.fit_report_.get("n_features"),
        }
