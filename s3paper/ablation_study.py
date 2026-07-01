"""Controlled ablation studies for S3-Forecaster and S3-FastSketch."""

from __future__ import annotations

import copy
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

from .metrics import evaluate_forecast
from .s3_fastsketch import S3FastSketchForecaster, _make_forecast_index
from .utils import count_trainable_parameters, ensure_series

class SimpleFoundationProxyAblation:
    def __init__(self, window_size=6):
        self.window_size = window_size

    def fit_predict(self, series, horizon):
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series).reshape(-1))

        trend = series.rolling(window=self.window_size, min_periods=1).mean().bfill()
        last_val = trend.iloc[-1]

        if isinstance(series.index, pd.DatetimeIndex):
            freq = pd.infer_freq(series.index)
            if freq is None:
                freq = "ME"
            future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
        else:
            future_index = pd.RangeIndex(start=len(series), stop=len(series) + horizon)

        detrended = series - trend

        if isinstance(series.index, pd.DatetimeIndex) and len(detrended) > 12:
            seasonality = detrended.groupby(detrended.index.month).mean()
            future_season = np.array(
                [seasonality[d.month] if d.month in seasonality.index else 0.0 for d in future_index],
                dtype=float
            )
        else:
            if len(detrended) > 12:
                seasonal_pattern = np.array(
                    [detrended.iloc[k::12].mean() for k in range(12)],
                    dtype=float
                )
            else:
                seasonal_pattern = np.full(12, float(detrended.mean()))

            last_pos = len(series) - 1
            future_season = np.array(
                [seasonal_pattern[(last_pos + h + 1) % 12] for h in range(horizon)],
                dtype=float
            )

        forecast = last_val + future_season
        return trend, pd.Series(forecast, index=future_index)

class EchoStateFeatureExtractorAblation:
    def __init__(self, reservoir_size=50, spectral_radius=0.9, seed=42):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.Win = None
        self.Wres = None
        self.rng = np.random.RandomState(seed)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def _init_weights(self):
        self.Win = self.rng.uniform(-1, 1, (self.reservoir_size, 1))
        W = self.rng.normal(0, 1, (self.reservoir_size, self.reservoir_size))
        eig = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eig))
        if max_eig == 0:
            max_eig = 1.0
        self.Wres = W * (self.spectral_radius / max_eig)

    def transform(self, residuals):
        residuals = np.asarray(residuals).reshape(-1)
        if self.Win is None:
            self._init_weights()

        self.scaler.fit(residuals.reshape(-1, 1))
        res_scaled = self.scaler.transform(residuals.reshape(-1, 1)).flatten()

        T = len(residuals)
        states = np.zeros((T, self.reservoir_size))
        h_t = np.zeros(self.reservoir_size)

        for t in range(T):
            h_t = np.tanh(np.dot(self.Win, [res_scaled[t]]) + np.dot(self.Wres, h_t))
            states[t, :] = h_t

        return states

class S3ForecasterAblation:
    """
    Modes:
      - base_only
      - base_residual
      - base_residual_gate
      - full_aci
      - full_static
    """

    def __init__(
        self,
        mode="full_aci",
        reservoir_size=50,
        spectral_radius=0.9,
        ar_lags=3,
        horizon=12,
        aci_step_size=0.05,
        foundation_window=6,
        regressor_type="Ridge",
        reg_alpha=1.0,
        oob_split_ratio=0.7,
        delta=0.1,
        seed=42,
    ):
        self.mode = mode
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.ar_lags = ar_lags
        self.horizon = horizon
        self.aci_step_size = aci_step_size
        self.foundation_window = foundation_window
        self.regressor_type = regressor_type
        self.reg_alpha = reg_alpha
        self.oob_split_ratio = oob_split_ratio
        self.delta = delta
        self.seed = seed

        self.foundation = SimpleFoundationProxyAblation(window_size=foundation_window)
        self.esn = EchoStateFeatureExtractorAblation(
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            seed=seed
        )

        self.readout = None
        self.gate_alpha = 1.0
        self.gate_beta = 0.0
        self.vol_stats = None
        self.conformity_scores = []
        self.current_alpha_t = delta
        self.static_q_width = None
        self.fitted = False

    def _build_readout(self):
        if self.regressor_type == "Ridge":
            base_reg = Ridge(alpha=self.reg_alpha)
        elif self.regressor_type == "Lasso":
            base_reg = Lasso(alpha=self.reg_alpha)
        elif self.regressor_type == "ElasticNet":
            base_reg = ElasticNet(alpha=self.reg_alpha, l1_ratio=0.5)
        else:
            base_reg = Ridge(alpha=self.reg_alpha)

        self.readout = MultiOutputRegressor(base_reg)

    def _compute_volatility_z(self, residuals):
        vol = pd.Series(residuals).ewm(span=4).std().fillna(0).values
        if self.vol_stats is None:
            iqr = np.percentile(vol, 75) - np.percentile(vol, 25)
            if iqr == 0:
                iqr = 1.0
            self.vol_stats = (np.median(vol), iqr + 1e-6)
        med, iqr = self.vol_stats
        return (vol - med) / (iqr / 1.35)

    def _get_gate(self, residuals, alpha, beta):
        z = self._compute_volatility_z(residuals)
        alpha = np.clip(alpha, 0.01, 50.0)
        return 1.0 / (1.0 + np.exp(-alpha * (z - beta)))

    def create_features(self, residuals):
        residuals = np.asarray(residuals).reshape(-1)
        esn = self.esn.transform(residuals)

        X_ar = np.zeros((len(residuals), self.ar_lags))
        for i in range(self.ar_lags, len(residuals)):
            X_ar[i, :] = residuals[i - self.ar_lags:i][::-1]

        return np.hstack([esn, X_ar])

    def _fit_readout_on_train_split(self, train_series):
        base_fit_tr, _ = self.foundation.fit_predict(train_series, self.horizon)
        res_tr = train_series.values - base_fit_tr.values

        X_tr = self.create_features(res_tr)
        Y_tr = np.zeros((len(res_tr), self.horizon)) * np.nan

        for i in range(len(res_tr) - self.horizon):
            Y_tr[i, :] = res_tr[i + 1:i + 1 + self.horizon]

        valid_mask = ~np.isnan(Y_tr).any(axis=1)
        valid_mask[:self.ar_lags] = False

        if np.sum(valid_mask) < 2:
            raise ValueError("Not enough valid samples to train residual adapter.")

        self._build_readout()
        self.readout.fit(X_tr[valid_mask], Y_tr[valid_mask])

    def _prepare_calibration(self, series, split_idx):
        full_base_fit, _ = self.foundation.fit_predict(series, self.horizon)
        full_res = series.values - full_base_fit.values

        X_full = self.create_features(full_res)
        calib_indices = np.arange(split_idx, len(series) - self.horizon)

        if len(calib_indices) == 0:
            return full_res, None, None, None, None

        X_calib = X_full[calib_indices]
        Y_calib = np.zeros((len(calib_indices), self.horizon))

        for idx, i in enumerate(calib_indices):
            Y_calib[idx, :] = full_res[i + 1:i + 1 + self.horizon]

        raw_preds_calib = self.readout.predict(X_calib)
        z_calib = self._compute_volatility_z(full_res)[calib_indices]

        return full_res, X_calib, Y_calib, raw_preds_calib, z_calib

    def _optimize_gate(self, Y_calib, raw_preds_calib, z_calib):
        def loss_fn(params):
            a, b = params
            a = np.clip(a, 0.01, 20.0)
            g = 1.0 / (1.0 + np.exp(-a * (z_calib - b)))
            g = g[:, np.newaxis]
            return np.mean(np.abs(Y_calib - raw_preds_calib * g))

        res_opt = minimize(loss_fn, [1.0, 0.0], bounds=[(0.1, 10), (-3, 3)])
        self.gate_alpha, self.gate_beta = res_opt.x

    def _fit_interval_block(self, Y_calib, raw_preds_calib, z_calib, use_gate):
        if use_gate:
            final_g = 1.0 / (1.0 + np.exp(-self.gate_alpha * (z_calib - self.gate_beta)))
            final_preds = raw_preds_calib * final_g[:, np.newaxis]
        else:
            final_preds = raw_preds_calib

        scores = np.abs(Y_calib - final_preds)
        self.conformity_scores = scores.flatten()

        if self.mode == "full_static":
            self.static_q_width = float(np.quantile(self.conformity_scores, 1.0 - self.delta))
            self.current_alpha_t = self.delta
            return

        # full_aci
        alpha_t = float(self.delta)
        for t in range(len(scores)):
            q_val = np.quantile(scores[:t + 1], 1.0 - alpha_t)
            is_covered = float(np.mean(scores[t] <= q_val))
            err = self.delta - (1.0 if is_covered > (1.0 - self.delta) else 0.0)
            alpha_t = alpha_t + self.aci_step_size * err
            alpha_t = float(np.clip(alpha_t, 0.01, 0.5))

        self.current_alpha_t = alpha_t

    def fit(self, series):
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series).reshape(-1))

        n = len(series)
        split_idx = int(n * self.oob_split_ratio)
        split_idx = max(split_idx, self.ar_lags + 2)
        split_idx = min(split_idx, n - self.horizon - 1)

        self.last_series = series

        # ----------------------------------------------------
        # base_only
        # ----------------------------------------------------
        if self.mode == "base_only":
            _, base_fit_full = self.foundation.fit_predict(series, self.horizon)
            self.last_residuals = series.values - self.foundation.fit_predict(series, self.horizon)[0].values
            self.fitted = True
            return self

        # ----------------------------------------------------
        # residual adapter and above
        # ----------------------------------------------------
        train_series = series.iloc[:split_idx]
        self._fit_readout_on_train_split(train_series)

        full_res, X_calib, Y_calib, raw_preds_calib, z_calib = self._prepare_calibration(series, split_idx)
        self.last_residuals = full_res

        if X_calib is None:
            self.fitted = True
            return self

        use_gate = self.mode in ["base_residual_gate", "full_aci", "full_static"]
        if use_gate:
            self._optimize_gate(Y_calib, raw_preds_calib, z_calib)

        if self.mode in ["full_aci", "full_static"]:
            self._fit_interval_block(Y_calib, raw_preds_calib, z_calib, use_gate=True)

        self.fitted = True
        return self

    def predict(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        _, base_forecast = self.foundation.fit_predict(self.last_series, self.horizon)
        pred_mean = base_forecast.values.copy()

        gate_val = 1.0

        if self.mode != "base_only":
            X_full = self.create_features(self.last_residuals)
            X_T = X_full[-1].reshape(1, -1)
            raw_pred = self.readout.predict(X_T)[0]

            if self.mode in ["base_residual_gate", "full_aci", "full_static"]:
                gate_val = float(self._get_gate(self.last_residuals, self.gate_alpha, self.gate_beta)[-1])
            else:
                gate_val = 1.0

            final_residual = raw_pred * gate_val
            pred_mean = pred_mean + final_residual

        # Point-only variants
        if self.mode in ["base_only", "base_residual", "base_residual_gate"]:
            return pd.DataFrame({
                "pred": pred_mean,
                "gate": gate_val
            }, index=base_forecast.index)

        # Interval variants
        if self.mode == "full_static":
            q_width = self.static_q_width
        else:
            q_width = float(np.quantile(self.conformity_scores, 1.0 - self.current_alpha_t))

        return pd.DataFrame({
            "pred": pred_mean,
            "lower": pred_mean - q_width,
            "upper": pred_mean + q_width,
            "gate": gate_val
        }, index=base_forecast.index)

def plot_s3_ablation_forecasts(train_series, test_series, forecasts, last_history=24):
    import matplotlib.pyplot as plt

    if not isinstance(train_series, pd.Series):
        train_series = pd.Series(np.asarray(train_series).reshape(-1))
    if not isinstance(test_series, pd.Series):
        test_series = pd.Series(np.asarray(test_series).reshape(-1))

    n = len(forecasts)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    titles = {
        "base_only": "base only",
        "base_residual": "base + residual adapter",
        "base_residual_gate": "base + residual + gate",
        "full_aci": "full model + ACI",
        "full_static": "full model + static conformal",
    }

    for ax, (mode, fcst) in zip(axes, forecasts.items()):
        ax.plot(train_series.index[-last_history:], train_series.values[-last_history:], label="history")
        ax.plot(test_series.index, test_series.values, label="real", color="green")
        ax.plot(fcst.index, fcst["pred"].values, label="forecast", color="red", linestyle="--")

        if {"lower", "upper"}.issubset(fcst.columns):
            ax.fill_between(fcst.index, fcst["lower"].values, fcst["upper"].values, alpha=0.18)

        ax.set_title(titles.get(mode, mode))
        ax.legend()

    plt.tight_layout()
    plt.show()

class ZeroFoundation:
    """Zero prior for the 'Without foundation' ablation."""
    def fit_predict(self, series, horizon):
        series = pd.Series(series).astype(float)
        fitted = pd.Series(np.zeros(len(series)), index=series.index)
        forecast = pd.Series(
            np.zeros(int(horizon)),
            index=_make_forecast_index(series, int(horizon)),
        )
        return fitted, forecast

class S3FastSketchAblation(S3FastSketchForecaster):
    """
    Controlled ablation without changing the original training protocol.

    Feature groups:
        ar, ema_residual, volatility, momentum,
        causal_conv, seasonal, calendar.
    """

    def __init__(
        self,
        *args,
        disabled_groups=None,
        use_foundation=True,
        foundation_only=False,
        use_shrinkage=True,
        use_aci=True,
        **kwargs,
    ):
        self.disabled_groups = set(disabled_groups or [])
        self.use_foundation_component = bool(use_foundation)
        self.foundation_only = bool(foundation_only)
        self.use_shrinkage = bool(use_shrinkage)
        self.use_aci = bool(use_aci)

        super().__init__(*args, **kwargs)

        if not self.use_foundation_component:
            self.foundation = ZeroFoundation()

    def create_features(self, residuals):
        X_all = super().create_features(residuals)
        names = list(self.feature_names_)
        groups = list(self.feature_groups_)

        keep = np.asarray(
            [group not in self.disabled_groups for group in groups],
            dtype=bool,
        )

        if keep.sum() == 0:
            self.feature_names_ = ["constant_zero"]
            self.feature_groups_ = ["constant"]
            return np.zeros((len(residuals), 1), dtype=float)

        self.feature_names_ = [n for n, k in zip(names, keep) if k]
        self.feature_groups_ = [g for g, k in zip(groups, keep) if k]
        return X_all[:, keep]

    def _recompute_conformal_state(self):
        """Recalibrate intervals after forcing gamma or disabling ACI."""
        n = len(self.last_series)
        split_idx = int(self.fit_report_["split_idx"])
        calib_indices = np.arange(split_idx, n - self.horizon)

        if len(calib_indices) == 0:
            self.conformity_scores = np.array([], dtype=float)
            self.current_alpha_t = float(self.aci_target)
            return

        y_calib = self._future_matrix(self.last_residuals_, calib_indices)
        raw_pred = self._safe_array(
            self.readout.predict(self.X_full_[calib_indices])
        )
        corrected = raw_pred * np.asarray(self.gamma_).reshape(1, -1)
        scores = np.max(np.abs(y_calib - corrected), axis=1)
        scores = scores[np.isfinite(scores)]

        buffer = []
        alpha_t = float(self.aci_target)

        for score in scores:
            score = float(score)

            if self.use_aci and len(buffer) >= 2:
                q = float(np.quantile(buffer, 1.0 - alpha_t))
                miss = int(score > q)
                alpha_t += self.aci_step_size * (self.aci_target - miss)
                alpha_t = float(np.clip(alpha_t, 0.01, 0.50))

            buffer.append(score)

        self.conformity_scores = np.asarray(buffer, dtype=float)
        self.current_alpha_t = (
            alpha_t if self.use_aci else float(self.aci_target)
        )

    def fit(self, series):
        super().fit(series)

        if not self.fitted:
            return self

        if self.foundation_only:
            self.gamma_ = np.zeros(self.horizon, dtype=float)
        elif not self.use_shrinkage:
            self.gamma_ = np.ones(self.horizon, dtype=float)

        if self.foundation_only or not self.use_shrinkage or not self.use_aci:
            self._recompute_conformal_state()

        return self


S3_ABLATION_LABELS = {
    "base_only": "base only",
    "base_residual": "base + residual adapter",
    "base_residual_gate": "base + residual + gate",
    "full_aci": "full model + ACI",
    "full_static": "full model + static conformal",
}

FASTSKETCH_ABLATIONS = {
    "Full": {},
    "Foundation only": {"foundation_only": True},
    "Without foundation": {"use_foundation": False},
    "Without AR lags": {"disabled_groups": {"ar"}},
    "Without EMA residual": {"disabled_groups": {"ema_residual"}},
    "Without volatility": {"disabled_groups": {"volatility"}},
    "Without momentum": {"disabled_groups": {"momentum"}},
    "Without causal convolutions": {"disabled_groups": {"causal_conv"}},
    "Without seasonal residual": {"disabled_groups": {"seasonal"}},
    "Without calendar": {"disabled_groups": {"calendar"}},
    "Without multiscale sketch": {
        "disabled_groups": {"ema_residual", "volatility", "momentum", "causal_conv"}
    },
    "Without signed shrinkage": {"use_shrinkage": False},
    "Static conformal (without ACI)": {"use_aci": False},
}


def run_s3_ablation_study(
    train_series: Any,
    test_series: Any,
    best_params: dict,
    *,
    alpha: float = 0.10,
    seasonal_period: int = 12,
    modes: Optional[list[str]] = None,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    modes = modes or list(S3_ABLATION_LABELS)
    rows, forecasts, models = [], {}, {}

    for mode in modes:
        start = time.perf_counter()
        model = S3ForecasterAblation(
            mode=mode,
            reservoir_size=best_params["reservoir_size"],
            spectral_radius=best_params["spectral_radius"],
            ar_lags=best_params["ar_lags"],
            horizon=len(test),
            aci_step_size=best_params["aci_step_size"],
            foundation_window=best_params["foundation_window"],
            regressor_type=best_params["regressor_type"],
            reg_alpha=best_params["reg_alpha"],
            oob_split_ratio=best_params["oob_split_ratio"],
            delta=alpha,
            seed=42,
        )
        try:
            model.fit(train)
            forecast = model.predict()
            elapsed = time.perf_counter() - start
            has_interval = {"lower", "upper"}.issubset(forecast.columns)
            metrics = evaluate_forecast(
                test,
                forecast["pred"],
                y_train=train if has_interval else None,
                lower=forecast["lower"] if has_interval else None,
                upper=forecast["upper"] if has_interval else None,
                alpha=alpha,
                seasonal_period=seasonal_period,
                elapsed_seconds=elapsed,
                trainable_params=count_trainable_parameters(getattr(model, "readout", None)),
            )
            row = {
                "variant": S3_ABLATION_LABELS.get(mode, mode),
                "mode": mode,
                "gate_value": float(forecast["gate"].iloc[0]) if "gate" in forecast else np.nan,
                "alpha_final": float(getattr(model, "current_alpha_t", np.nan)),
                "status": "ok",
                "error": None,
                **metrics,
            }
            forecasts[mode], models[mode] = forecast, model
        except Exception as exc:
            row = {
                "variant": S3_ABLATION_LABELS.get(mode, mode),
                "mode": mode,
                "status": "failed",
                "error": str(exc),
            }
            forecasts[mode], models[mode] = None, None
        rows.append(row)
    return {"results": pd.DataFrame(rows), "forecasts": forecasts, "models": models}


def run_fastsketch_ablation(
    train_series: Any,
    test_series: Any,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    variants: Optional[dict] = None,
    series_id: str = "series",
    seasonal_period: int = 12,
    alpha: float = 0.10,
    verbose: bool = False,
):
    from .metrics import seasonal_naive_scale

    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    uq = dict(uq_params or {})
    variants = copy.deepcopy(variants or FASTSKETCH_ABLATIONS)
    point_keys = {
        "foundation_window", "ar_lags", "ema_spans", "conv_scales",
        "use_calendar", "ridge_alpha", "oob_split_ratio", "shrinkage_max",
    }
    model_params = {k: v for k, v in point_params.items() if k in point_keys}
    min_width = float(uq.get("min_width_factor", 0.0)) * seasonal_naive_scale(
        train, seasonal_period=seasonal_period
    )

    rows, forecasts, models = [], {}, {}
    for variant_name, kwargs in variants.items():
        start = time.perf_counter()
        try:
            model = S3FastSketchAblation(
                horizon=1,
                seasonal_period=seasonal_period,
                aci_target=float(uq.get("aci_target", alpha)),
                aci_step_size=float(uq.get("aci_step_size", 0.05)),
                min_train_samples=6,
                min_calib_samples=2,
                **model_params,
                **kwargs,
            )
            model.fit(train)
            if not model.fitted:
                raise RuntimeError(model.fit_report_.get("reason", "fit failed"))
            forecast = model.predict(
                steps=len(test),
                recursive=True,
                interval_scale=float(uq.get("interval_scale", 1.0)),
                interval_power=float(uq.get("interval_power", 0.5)),
                min_width=min_width,
            )
            elapsed = time.perf_counter() - start
            metrics = evaluate_forecast(
                test,
                forecast["pred"],
                y_train=train,
                lower=forecast["lower"],
                upper=forecast["upper"],
                alpha=alpha,
                seasonal_period=seasonal_period,
                elapsed_seconds=elapsed,
                trainable_params=0 if model.foundation_only else count_trainable_parameters(model),
            )
            row = {
                "series_id": series_id,
                "variant": variant_name,
                "n_features": 0 if model.foundation_only else model.fit_report_.get("n_features", np.nan),
                "gamma": tuple(np.asarray(model.gamma_, dtype=float)),
                "final_alpha_t": float(model.current_alpha_t),
                "status": "ok",
                "error": None,
                **metrics,
            }
            forecasts[variant_name], models[variant_name] = forecast, model
            if verbose:
                print(f"[OK] {variant_name}: MAPE={metrics['mape_percent']:.3f}%, MSIS={metrics['msis']:.4f}")
        except Exception as exc:
            row = {"series_id": series_id, "variant": variant_name, "status": "failed", "error": str(exc)}
            forecasts[variant_name], models[variant_name] = None, None
            if verbose:
                print(f"[FAILED] {variant_name}: {exc}")
        rows.append(row)

    results = pd.DataFrame(rows)
    full = results[(results["variant"] == "Full") & (results["status"] == "ok")]
    if len(full) == 1:
        baseline = full.iloc[0]
        for metric in ["mape_percent", "rmse", "coverage_error", "mean_width", "msis"]:
            if metric in results:
                results[f"delta_{metric}"] = results[metric] - baseline[metric]
    return {"results": results, "forecasts": forecasts, "models": models}


def run_fastsketch_ablation_dataset(
    train_series_map: dict,
    test_series_map: dict,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    variants: Optional[dict] = None,
    series_ids: Optional[list] = None,
    max_series: Optional[int] = None,
    log_if_positive: bool = True,
    seasonal_period: int = 12,
    alpha: float = 0.10,
):
    ids = [sid for sid in train_series_map if sid in test_series_map]
    if series_ids is not None:
        selected = set(series_ids)
        ids = [sid for sid in ids if sid in selected]
    if max_series is not None:
        ids = ids[: int(max_series)]

    details, transforms = [], []
    for sid in ids:
        train = ensure_series(train_series_map[sid])
        test = ensure_series(test_series_map[sid])
        transformed = bool(log_if_positive and (train > 0).all() and (test > 0).all())
        if transformed:
            train = pd.Series(np.log(train.to_numpy()), index=train.index, name=train.name)
            test = pd.Series(np.log(test.to_numpy()), index=test.index, name=test.name)
        out = run_fastsketch_ablation(
            train, test, point_params, uq_params=uq_params, variants=variants,
            series_id=str(sid), seasonal_period=seasonal_period, alpha=alpha,
        )
        details.append(out["results"])
        transforms.append({"series_id": sid, "log_transformed": transformed, "n_train": len(train), "n_test": len(test)})

    detailed = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    valid = detailed[detailed.get("status") == "ok"].copy() if len(detailed) else detailed
    metric_cols = [c for c in ["mape_percent", "smape_percent", "mae", "rmse", "r2", "ecp", "coverage_error", "mean_width", "interval_score", "msis", "n_features", "trainable_params"] if c in valid]
    if len(valid):
        aggregate = pd.concat([
            valid.groupby("variant")[metric_cols].mean().add_suffix("_mean"),
            valid.groupby("variant")[metric_cols].median().add_suffix("_median"),
            valid.groupby("variant").size().rename("n_series"),
        ], axis=1).reset_index()
    else:
        aggregate = pd.DataFrame()
    return {"detailed": detailed, "aggregate": aggregate, "transformations": pd.DataFrame(transforms)}


def paired_ablation_wilcoxon(detailed_results: pd.DataFrame, metric: str = "mape_percent", baseline: str = "Full"):
    valid = detailed_results[detailed_results["status"] == "ok"]
    pivot = valid.pivot_table(index="series_id", columns="variant", values=metric, aggfunc="first")
    rows = []
    if baseline not in pivot:
        return pd.DataFrame()
    for variant in pivot.columns:
        if variant == baseline:
            continue
        pair = pivot[[baseline, variant]].dropna()
        delta = pair[variant].to_numpy() - pair[baseline].to_numpy()
        if len(delta) == 0:
            statistic, p_value = np.nan, np.nan
        elif np.allclose(delta, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(pair[variant], pair[baseline])
        rows.append({
            "metric": metric, "variant": variant, "n_pairs": len(pair),
            "baseline_mean": pair[baseline].mean(), "ablation_mean": pair[variant].mean(),
            "mean_delta": np.mean(delta) if len(delta) else np.nan,
            "wilcoxon_statistic": statistic, "p_value": p_value,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    finite = result["p_value"].notna()
    ordered = result.loc[finite].sort_values("p_value").index.tolist()
    adjusted = pd.Series(np.nan, index=result.index)
    running = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        corrected = min(1.0, (m - rank) * result.loc[idx, "p_value"])
        running = max(running, corrected)
        adjusted.loc[idx] = running
    result["p_holm"] = adjusted
    result["significant_0.05"] = result["p_holm"] < 0.05
    return result.sort_values("p_holm")
