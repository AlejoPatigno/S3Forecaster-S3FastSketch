"""S3-FastSketch: fast causal residual sketching for small-data forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from .s3_forecaster import SimpleFoundationProxy


def _make_forecast_index(series, horizon):
    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is not None:
            offset = pd.tseries.frequencies.to_offset(freq)
            return pd.date_range(start=idx[-1] + offset, periods=horizon, freq=freq)
    if isinstance(idx, pd.PeriodIndex):
        freq = idx.freqstr or "M"
        return pd.period_range(start=idx[-1] + 1, periods=horizon, freq=freq)
    return pd.RangeIndex(start=len(series), stop=len(series) + horizon)

class FastRollingFoundation:
    """
    Prior causal rápido: media móvil.
    Puede reemplazarse por tu SimpleFoundationProxy si ya lo tienes.
    """

    def __init__(self, window_size=6):
        self.window_size = int(window_size)

    def fit_predict(self, series, horizon):
        y = np.asarray(series.values, dtype=float)
        n = len(y)

        fitted = np.zeros(n, dtype=float)

        for t in range(n):
            if t == 0:
                fitted[t] = y[0]
            else:
                start = max(0, t - self.window_size)
                fitted[t] = np.mean(y[start:t])

        hist = list(y.copy())
        fcst = []

        for _ in range(horizon):
            start = max(0, len(hist) - self.window_size)
            pred = float(np.mean(hist[start:]))
            fcst.append(pred)
            hist.append(pred)

        fitted = pd.Series(fitted, index=series.index)
        forecast = pd.Series(fcst, index=_make_forecast_index(series, horizon))

        return fitted, forecast

class S3FastSketchForecaster:
    """
    S3-FastSketch:
    Complemento rápido de S3Forecaster basado en residual sketching causal.

    Diferencias frente a S3ForecasterV6:
    - No usa ESN recurrente.
    - No prueba múltiples priors.
    - No optimiza gate con scipy.minimize.
    - Extrae información paralela mediante filtros causales fijos.
    - Usa Ridge multi-output + shrinkage cerrado por calibración.
    - Mantiene ACI para intervalos predictivos.
    """

    def __init__(
        self,
        horizon=1,
        foundation_window=6,
        ar_lags=3,
        ema_spans=(2, 4, 8, 12),
        conv_scales=(2, 3, 4, 6),
        seasonal_period=12,
        use_calendar=True,
        ridge_alpha=1.0,
        oob_split_ratio=0.64,
        shrinkage_max=1.5,
        aci_target=0.10,
        aci_step_size=0.05,
        min_train_samples=8,
        min_calib_samples=2,
        use_existing_simple_foundation=True
    ):
        self.horizon = int(horizon)
        self.foundation_window = int(foundation_window)
        self.ar_lags = int(ar_lags)

        self.ema_spans = tuple(ema_spans)
        self.conv_scales = tuple(conv_scales)
        self.seasonal_period = int(seasonal_period)
        self.use_calendar = bool(use_calendar)

        self.ridge_alpha = float(ridge_alpha)
        self.oob_split_ratio = float(oob_split_ratio)
        self.shrinkage_max = float(shrinkage_max)

        self.aci_target = float(aci_target)
        self.aci_step_size = float(aci_step_size)

        self.min_train_samples = int(min_train_samples)
        self.min_calib_samples = int(min_calib_samples)
        self.use_existing_simple_foundation = bool(use_existing_simple_foundation)

        self.foundation = self._make_foundation()
        self.readout = MultiOutputRegressor(Ridge(alpha=self.ridge_alpha))

        self.fitted = False
        self.fit_report_ = {}
        self.conformity_scores = np.array([])
        self.current_alpha_t = self.aci_target
        self.gamma_ = None
        self.feature_names_ = None
        self.feature_groups_ = None

    # -------------------------------------------------------------------------
    # Prior
    # -------------------------------------------------------------------------

    def _make_foundation(self):
        if self.use_existing_simple_foundation:
            return SimpleFoundationProxy(window_size=self.foundation_window)

        return FastRollingFoundation(window_size=self.foundation_window)

    # -------------------------------------------------------------------------
    # Utilidades
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_series(series):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        series = series.astype(float)
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.dropna()

        return series

    @staticmethod
    def _safe_array(x):
        return np.nan_to_num(
            np.asarray(x, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    def _future_matrix(self, values, indices):
        values = np.asarray(values, dtype=float)
        Y = np.full((len(indices), self.horizon), np.nan)

        for k, i in enumerate(indices):
            Y[k, :] = values[i + 1:i + 1 + self.horizon]

        return Y

    # -------------------------------------------------------------------------
    # Sketch causal paralelo
    # -------------------------------------------------------------------------

    def _causal_conv_last(self, x, t, scale, kind):
        """
        Filtros causales pequeños.
        No se entrenan. Funcionan como detectores rápidos de forma local.
        """
        start = max(0, t - scale + 1)
        window = x[start:t + 1]

        if len(window) == 0:
            return 0.0

        if kind == "mean":
            return float(np.mean(window))

        if kind == "slope":
            if len(window) < 2:
                return 0.0
            return float(window[-1] - window[0])

        if kind == "contrast":
            if len(window) < 2:
                return 0.0
            mid = len(window) // 2
            left = window[:mid]
            right = window[mid:]
            if len(left) == 0 or len(right) == 0:
                return 0.0
            return float(np.mean(right) - np.mean(left))

        if kind == "energy":
            return float(np.mean(np.abs(window)))

        return 0.0

    def create_features(self, residuals):
        """
        Construye características en cada tiempo t usando solo información <= t.
        """
        e = self._safe_array(residuals).reshape(-1)
        n = len(e)

        features = []
        names = []
        groups = []

        # ---------------------------------------------------------------------
        # 1. Lags AR del residual
        # ---------------------------------------------------------------------
        for lag in range(1, self.ar_lags + 1):
            col = np.zeros(n)
            col[lag:] = e[:-lag]
            features.append(col)
            names.append(f"ar_lag_{lag}")
            groups.append("ar")

        # ---------------------------------------------------------------------
        # 2. EMA multiescala del residual
        # ---------------------------------------------------------------------
        for span in self.ema_spans:
            col = (
                pd.Series(e)
                .ewm(span=span, adjust=False)
                .mean()
                .fillna(0.0)
                .values
            )
            features.append(col)
            names.append(f"ema_resid_{span}")
            groups.append("ema_residual")

        # ---------------------------------------------------------------------
        # 3. EMA multiescala de magnitud absoluta: shock/volatility proxy
        # ---------------------------------------------------------------------
        abs_e = np.abs(e)

        for span in self.ema_spans:
            col = (
                pd.Series(abs_e)
                .ewm(span=span, adjust=False)
                .mean()
                .fillna(0.0)
                .values
            )
            features.append(col)
            names.append(f"ema_abs_{span}")
            groups.append("volatility")

        # ---------------------------------------------------------------------
        # 4. Momentum / diferencias locales
        # ---------------------------------------------------------------------
        for span in self.ema_spans:
            diff = np.zeros(n)
            if n > span:
                diff[span:] = e[span:] - e[:-span]
            features.append(diff)
            names.append(f"diff_{span}")
            groups.append("momentum")

        # ---------------------------------------------------------------------
        # 5. Filtros convolucionales causales fijos
        # ---------------------------------------------------------------------
        conv_kinds = ["mean", "slope", "contrast", "energy"]

        for scale in self.conv_scales:
            for kind in conv_kinds:
                col = np.zeros(n)
                for t in range(n):
                    col[t] = self._causal_conv_last(e, t, scale, kind)

                features.append(col)
                names.append(f"conv_{kind}_{scale}")
                groups.append("causal_conv")

        # ---------------------------------------------------------------------
        # 6. Señales estacionales del residual
        # ---------------------------------------------------------------------
        m = self.seasonal_period

        if m is not None and m > 1:
            seasonal_lag = np.zeros(n)
            if n > m:
                seasonal_lag[m:] = e[:-m]

            seasonal_gap = np.zeros(n)
            if n > m:
                seasonal_gap[m:] = e[m:] - e[:-m]

            features.append(seasonal_lag)
            names.append(f"seasonal_lag_{m}")
            groups.append("seasonal")

            features.append(seasonal_gap)
            names.append(f"seasonal_gap_{m}")
            groups.append("seasonal")

        # ---------------------------------------------------------------------
        # 7. Calendario armónico simple
        # ---------------------------------------------------------------------
        if self.use_calendar and m is not None and m > 1:
            t = np.arange(n)
            sin_col = np.sin(2.0 * np.pi * t / m)
            cos_col = np.cos(2.0 * np.pi * t / m)

            features.append(sin_col)
            names.append(f"sin_period_{m}")
            groups.append("calendar")

            features.append(cos_col)
            names.append(f"cos_period_{m}")
            groups.append("calendar")

        X = np.column_stack(features) if len(features) > 0 else np.zeros((n, 1))
        X = self._safe_array(X)

        # Normalización robusta por columna.
        med = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr <= 1e-8] = 1.0

        X = (X - med) / (iqr / 1.35 + 1e-8)
        X = self._safe_array(X)

        self.feature_names_ = names
        self.feature_groups_ = groups

        return X

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(self, series):
        series = self._clean_series(series)
        n = len(series)

        self.fitted = False
        self.last_series = series.copy()
        self.conformity_scores = np.array([])
        self.current_alpha_t = self.aci_target
        self.gamma_ = None

        self.fit_report_ = {
            "n": n,
            "horizon": self.horizon,
            "reason": None,
            "split_idx": None,
            "n_features": None,
            "n_train_readout": None,
            "n_calib": None
        }

        min_required = (
            self.ar_lags
            + self.horizon
            + self.min_train_samples
            + self.min_calib_samples
        )

        if n < min_required:
            self.fit_report_["reason"] = (
                f"Insufficient data: n={n}, required at least {min_required}. "
                f"Use horizon=1, reduce ar_lags, or provide more data."
            )
            return self

        split_idx = int(n * self.oob_split_ratio)

        min_split = self.ar_lags + self.horizon + self.min_train_samples
        max_split = n - self.horizon - self.min_calib_samples

        if max_split <= min_split:
            self.fit_report_["reason"] = (
                f"Invalid split: min_split={min_split}, max_split={max_split}."
            )
            return self

        split_idx = int(np.clip(split_idx, min_split, max_split))
        self.fit_report_["split_idx"] = split_idx

        train_series = series.iloc[:split_idx]

        # ---------------------------------------------------------------------
        # Base prior y residuales
        # ---------------------------------------------------------------------
        base_fit_train, _ = self.foundation.fit_predict(train_series, self.horizon)
        base_fit_train = base_fit_train.reindex(train_series.index).ffill().bfill()

        residual_train = train_series.values - base_fit_train.values
        residual_train = self._safe_array(residual_train)

        X_train_full = self.create_features(residual_train)
        Y_train_full = self._future_matrix(
            residual_train,
            np.arange(0, len(residual_train) - self.horizon)
        )

        valid_indices = np.arange(0, len(residual_train) - self.horizon)
        valid_mask = np.ones(len(valid_indices), dtype=bool)

        valid_mask[:self.ar_lags] = False
        valid_mask = valid_mask & np.all(np.isfinite(Y_train_full), axis=1)

        if np.sum(valid_mask) < self.min_train_samples:
            self.fit_report_["reason"] = (
                f"Not enough valid readout samples: {int(np.sum(valid_mask))}."
            )
            return self

        X_fit = X_train_full[valid_indices[valid_mask]]
        Y_fit = Y_train_full[valid_mask]

        self.readout = MultiOutputRegressor(Ridge(alpha=self.ridge_alpha))
        self.readout.fit(X_fit, Y_fit)

        # ---------------------------------------------------------------------
        # Full residuals para calibración
        # ---------------------------------------------------------------------
        full_base_fit, _ = self.foundation.fit_predict(series, self.horizon)
        full_base_fit = full_base_fit.reindex(series.index).ffill().bfill()

        full_residual = series.values - full_base_fit.values
        full_residual = self._safe_array(full_residual)

        X_full = self.create_features(full_residual)

        calib_indices = np.arange(split_idx, n - self.horizon)

        if len(calib_indices) < self.min_calib_samples:
            self.fit_report_["reason"] = (
                f"Calibration block too small: {len(calib_indices)}."
            )
            return self

        Y_calib = self._future_matrix(full_residual, calib_indices)
        raw_resid_pred = self.readout.predict(X_full[calib_indices])

        raw_resid_pred = self._safe_array(raw_resid_pred)
        Y_calib = self._safe_array(Y_calib)

        # ---------------------------------------------------------------------
        # MAPE-calibrated signed shrinkage
        # ---------------------------------------------------------------------
        actual_y_calib = self._future_matrix(series.values, calib_indices)
        
        gamma_grid = np.linspace(
            -self.shrinkage_max,
            self.shrinkage_max,
            81
        )
        
        gamma = np.zeros(self.horizon)
        
        for h in range(self.horizon):
            base_h = self._future_matrix(full_base_fit.values, calib_indices)[:, h]
            resid_pred_h = raw_resid_pred[:, h]
            y_true_h = actual_y_calib[:, h]
        
            best_gamma = 0.0
            best_score = np.inf
        
            for g in gamma_grid:
                y_pred_h = base_h + g * resid_pred_h
        
                score = np.mean(
                    np.abs(y_true_h - y_pred_h) /
                    np.maximum(np.abs(y_true_h), 1e-5)
                )
        
                if score < best_score:
                    best_score = score
                    best_gamma = g
    
            gamma[h] = best_gamma
        
        self.gamma_ = gamma
        
        calib_resid_corrected = raw_resid_pred * gamma.reshape(1, -1)

        # ---------------------------------------------------------------------
        # ACI sobre error final residual
        # ---------------------------------------------------------------------
        score_matrix = np.abs(Y_calib - calib_resid_corrected)

        buffer = []
        alpha_t = self.aci_target

        for i in range(len(score_matrix)):
            current_score = float(np.max(score_matrix[i]))

            if len(buffer) >= 2:
                q_val = np.quantile(buffer, 1.0 - alpha_t)
                err_t = 1 if current_score > q_val else 0

                alpha_t = alpha_t + self.aci_step_size * (self.aci_target - err_t)
                alpha_t = float(np.clip(alpha_t, 0.01, 0.50))

            buffer.append(current_score)

        self.conformity_scores = np.asarray(buffer, dtype=float)
        self.current_alpha_t = alpha_t

        self.last_residuals_ = full_residual
        self.X_full_ = X_full
        self.full_base_fit_ = full_base_fit

        self.fit_report_["reason"] = "success"
        self.fit_report_["n_features"] = X_full.shape[1]
        self.fit_report_["n_train_readout"] = int(np.sum(valid_mask))
        self.fit_report_["n_calib"] = int(len(calib_indices))

        self.fitted = True

        return self

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    def predict(
        self,
        steps=None,
        recursive=True,
        interval_scale=1.0,
        interval_power=0.5,
        min_width=0.0
    ):
        """
        Predicción directa o recursiva con calibración explícita de intervalos.
    
        Parámetros
        ----------
        steps : int or None
            Número de pasos futuros.
    
        recursive : bool
            Si True, genera predicción one-step recursiva.
    
        interval_scale : float
            Multiplicador global del ancho conformal.
    
        interval_power : float
            Exponente de crecimiento multi-step.
            p=0.0: ancho constante.
            p=0.5: crecimiento sqrt(h).
            p=1.0: crecimiento lineal.
    
        min_width : float
            Ancho mínimo absoluto del intervalo.
        """

        if not hasattr(self, "fitted") or not self.fitted:
            reason = getattr(self, "fit_report_", {}).get("reason", "Unknown reason.")
            raise Exception(f"Model fit failed. Reason: {reason}")
    
        if steps is None:
            steps = self.horizon
    
        steps = int(steps)
    
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")
    
        interval_scale = float(interval_scale)
        interval_power = float(interval_power)
        min_width = float(min_width)
    
        if len(self.conformity_scores) > 0:
            q_base = float(
                np.quantile(
                    self.conformity_scores,
                    1.0 - self.current_alpha_t
                )
            )
        else:
            q_base = 0.0

        def width_at_step(h):
            q = interval_scale * q_base * (h ** interval_power)
            q = max(q, min_width)
            return float(q)
    
        # -------------------------------------------------------------------------
        # Direct multi-horizon mode
        # -------------------------------------------------------------------------
        if not recursive:
            if steps > self.horizon:
                raise ValueError(
                    f"Direct prediction only supports steps <= self.horizon. "
                    f"Received steps={steps}, self.horizon={self.horizon}. "
                    f"Use recursive=True."
                )
    
            _, base_forecast = self.foundation.fit_predict(
                self.last_series,
                self.horizon
            )
    
            X_T = self.X_full_[-1].reshape(1, -1)
    
            raw_resid_pred = self.readout.predict(X_T)[0]
            raw_resid_pred = self._safe_array(raw_resid_pred)
    
            corrected_resid = raw_resid_pred * self.gamma_
    
            pred = base_forecast.values + corrected_resid
    
            pred = pred[:steps]
            base_values = base_forecast.values[:steps]
            raw_resid_pred = raw_resid_pred[:steps]
            corrected_resid = corrected_resid[:steps]
            gamma_used = self.gamma_[:steps]
    
            q_widths = np.array([width_at_step(h) for h in range(1, steps + 1)])
    
            out = pd.DataFrame({
                "step": np.arange(1, steps + 1),
                "base": base_values,
                "pred": pred,
                "lower": pred - q_widths,
                "upper": pred + q_widths,
                "residual_raw": raw_resid_pred,
                "residual_corrected": corrected_resid,
                "gamma": gamma_used,
                "q_width": q_widths
            }, index=base_forecast.index[:steps])
    
            return out
    
        # -------------------------------------------------------------------------
        # Recursive one-step-ahead mode
        # -------------------------------------------------------------------------
        current_series = self.last_series.copy()
    
        rows = []
        forecast_index = []
    
        for h in range(1, steps + 1):
    
            full_base_fit, base_forecast = self.foundation.fit_predict(
                current_series,
                horizon=1
            )
    
            full_base_fit = (
                full_base_fit
                .reindex(current_series.index)
                .ffill()
                .bfill()
            )
    
            base_next = float(base_forecast.values[0])
            next_idx = base_forecast.index[0]
    
            residuals = current_series.values - full_base_fit.values
            residuals = self._safe_array(residuals)
    
            X_current = self.create_features(residuals)
            X_T = X_current[-1].reshape(1, -1)
    
            raw_pred = self.readout.predict(X_T)[0]
            raw_pred = self._safe_array(raw_pred)
    
            raw_resid_1 = float(raw_pred[0])
    
            gamma_1 = float(self.gamma_[0]) if self.gamma_ is not None else 0.0
            corrected_resid_1 = gamma_1 * raw_resid_1
    
            y_pred = base_next + corrected_resid_1
    
            q_width = width_at_step(h)
    
            rows.append({
                "step": h,
                "base": base_next,
                "pred": y_pred,
                "lower": y_pred - q_width,
                "upper": y_pred + q_width,
                "residual_raw": raw_resid_1,
                "residual_corrected": corrected_resid_1,
                "gamma": gamma_1,
                "q_width": q_width
            })
    
            forecast_index.append(next_idx)
    
            current_series.loc[next_idx] = y_pred
    
            try:
                current_series = current_series.sort_index()
            except Exception:
                pass
    
        return pd.DataFrame(rows, index=forecast_index)

    # -------------------------------------------------------------------------
    # Interpretabilidad rápida
    # -------------------------------------------------------------------------

    def feature_importance(self):
        if not self.fitted:
            return pd.DataFrame()

        rows = []

        for h, estimator in enumerate(self.readout.estimators_):
            coef = np.asarray(estimator.coef_).reshape(-1)

            for name, group, value in zip(
                self.feature_names_,
                self.feature_groups_,
                coef
            ):
                rows.append({
                    "horizon": h + 1,
                    "feature": name,
                    "group": group,
                    "coef": value,
                    "abs_coef": abs(value)
                })

        return pd.DataFrame(rows).sort_values(
            ["horizon", "abs_coef"],
            ascending=[True, False]
        )

    def group_importance(self):
        fi = self.feature_importance()

        if fi.empty:
            return pd.DataFrame()

        return (
            fi.groupby(["horizon", "group"], as_index=False)["abs_coef"]
            .sum()
            .sort_values(["horizon", "abs_coef"], ascending=[True, False])
        )

    def summary(self):
        return {
            "fitted": self.fitted,
            "fit_report": self.fit_report_,
            "gamma": self.gamma_,
            "current_alpha_t": self.current_alpha_t,
            "n_conformity_scores": len(self.conformity_scores),
            "n_features": self.fit_report_.get("n_features", None)
        }

def _fastsketch_trainable_parameter_count(self):
    total = 0
    for estimator in getattr(self.readout, "estimators_", []):
        total += np.asarray(estimator.coef_).size
        total += np.asarray(estimator.intercept_).size
    if getattr(self, "gamma_", None) is not None:
        total += np.asarray(self.gamma_).size
    return int(total)


S3FastSketchForecaster.trainable_parameter_count = _fastsketch_trainable_parameter_count
