# Cambios funcionales requeridos para habilitar la regeneración de resultados del artículo

**Repositorio:** `AlejoPatigno/S3Forecaster-S3FastSketch`  
**SHA auditado:** `33b1bcf38d3828f20bacfac9b91dcd904c32a946`  
**Objetivo:** dejar el paquete metodológico listo para ejecutar nuevamente los experimentos del artículo sin inconsistencias entre HPO, predicción puntual, calibración conformal, intervalos reportados y métricas.

---

## 1. Alcance de este documento

Este documento se limita a cambios dentro del código Python del paquete. Quedan explícitamente fuera de alcance:

- notebooks;
- ejecución en Kaggle;
- carga o descarga de datasets;
- implementación de loaders específicos;
- generación material de tablas, figuras o resultados;
- ejecución final de los experimentos.

El objetivo aquí es corregir las funciones que todavía impedirían que una futura regeneración de resultados sea metodológicamente válida.

---

## 2. Prioridad de implementación

### Bloqueadores P0

1. Integrar `interval_scale` y `minimum_width` dentro del ciclo online de ACI.
2. Eliminar el escalado de intervalos posterior al rolling forecast.
3. Recalcular siempre las métricas a partir del forecast final retornado.
4. Propagar correctamente `seasonal_period` en HPO, evaluación y priors.
5. Reescribir el HPO multiserie para que use directamente el `trial` externo de Optuna.
6. Implementar HPO UQ multiserie con parámetros point congelados.
7. Evitar que las predicciones recursivas actualicen ACI con pseudobservaciones.

### Cambios P1

8. Unificar el tratamiento de intervalos entre S3-Forecaster, S3-FastSketch y baselines.
9. Hacer que el HPO uniserie delegue en el mismo motor de temporal CV utilizado por el HPO multiserie.
10. Añadir pruebas de consistencia entre forecast, ACI y métricas.

### Cambios P2

11. Integrar el cache de priors en los recorridos causales costosos.
12. Generalizar el wrapper de transformaciones para baselines como ARIMA.
13. Consolidar el manejo de fallos y metadatos de HPO.

---

# 3. Corrección del intervalo online

## 3.1. Problema actual

En `s3_forecaster_experiment.py` y `s3_fastsketch_experiment.py`, el intervalo se escala después de ejecutar `evaluate_rolling_model()`.

Esto provoca que:

- ACI determine el error de cobertura usando un intervalo;
- `alpha_t` se actualice usando ese intervalo;
- el forecast final reporte otro intervalo;
- las métricas puedan calcularse con límites diferentes a los usados durante la adaptación online.

En S3-Forecaster existe además un error adicional: después de modificar los límites, se conservan las métricas originales mediante `metrics = result["metrics"]`.

## 3.2. Cambio en `s3paper/conformal.py`

Añadir una función común para finalizar intervalos simétricos:

```python
def finalize_symmetric_interval(
    point_prediction: float,
    raw_interval: tuple[float, float],
    *,
    interval_scale: float = 1.0,
    minimum_width: float = 0.0,
) -> tuple[float, float, float]:
    point = float(point_prediction)
    lower_raw, upper_raw = map(float, raw_interval)

    raw_half_width = max(point - lower_raw, upper_raw - point, 0.0)
    final_half_width = max(
        float(interval_scale) * raw_half_width,
        float(minimum_width),
    )

    lower = point - final_half_width
    upper = point + final_half_width
    return lower, upper, final_half_width
```

### Validaciones obligatorias

La función debe rechazar:

```python
interval_scale <= 0
minimum_width < 0
```

También debe asegurar:

```python
lower <= point_prediction <= upper
upper - point_prediction == point_prediction - lower
```

---

# 4. Cambios en `S3Forecaster`

Archivo: `s3paper/s3_forecaster.py`

## 4.1. Modificar `S3Forecaster.__init__`

Añadir:

```python
interval_scale: float = 1.0,
minimum_width: float = 0.0,
```

Guardar:

```python
self.interval_scale = float(interval_scale)
self.minimum_width = float(minimum_width)
```

Validar los valores en el constructor.

## 4.2. Modificar `S3Forecaster.predict_one`

Reemplazar la generación directa:

```python
lower, upper = self.aci_.interval(pred)
```

por:

```python
raw_interval = self.aci_.interval(pred) if self.use_aci else (pred, pred)
lower, upper, half_width = finalize_symmetric_interval(
    pred,
    raw_interval,
    interval_scale=self.interval_scale,
    minimum_width=self.minimum_width,
)
```

Guardar el intervalo final en `last_prediction_`:

```python
self.last_prediction_ = {
    "base": base,
    "pred": pred,
    "raw": raw,
    "gate": gate,
    "lower": lower,
    "upper": upper,
    "half_width": half_width,
}
```

La columna `q_width` debe representar claramente una de estas dos cantidades y mantener la misma convención en todo el repositorio:

- semi-anchura: `upper - pred`; o
- anchura total: `upper - lower`.

Recomendación: renombrar a `half_width` para evitar ambigüedad.

## 4.3. Mantener `S3Forecaster.update` usando el intervalo final

La función ya utiliza `last_prediction_["lower"]` y `last_prediction_["upper"]`. Después del cambio anterior, esos límites serán exactamente los límites reportados.

No debe reconstruirse el intervalo dentro de `update()`.

## 4.4. Corregir `S3Forecaster.predict`

Actualmente la predicción recursiva llama:

```python
self.update(float(row["pred"].iloc[0]))
```

Esto hace que ACI trate una predicción como si fuera una observación verdadera.

Modificar `update` para aceptar:

```python
def update(self, observation: float, *, is_observed: bool = True):
```

Aplicar:

```python
if is_observed and self.last_prediction_ is not None:
    self.aci_.update(...)
```

En predicción recursiva:

```python
self.update(predicted_value, is_observed=False)
```

Para el protocolo principal de horizonte uno, `rolling_one_step_forecast()` debe continuar usando:

```python
model.update(real_observation, is_observed=True)
```

### Criterio metodológico

Las pseudobservaciones pueden actualizar el estado recursivo del prior, pero nunca deben:

- añadir conformity scores;
- modificar `alpha_t`;
- contabilizar cobertura.

---

# 5. Cambios en `S3FastSketchForecaster`

Archivo: `s3paper/s3_fastsketch.py`

## 5.1. Modificar `S3FastSketchForecaster.__init__`

Añadir los mismos parámetros:

```python
interval_scale: float = 1.0,
minimum_width: float = 0.0,
```

## 5.2. Modificar `predict_one`

Aplicar `finalize_symmetric_interval()` inmediatamente después de `self.aci_.interval(pred)` y antes de asignar `last_prediction_`.

El intervalo almacenado, reportado y posteriormente entregado a ACI debe ser idéntico.

## 5.3. Eliminar el escalado posterior dentro de `predict`

La función actual recibe:

```python
interval_scale
interval_power
min_width
```

y modifica el DataFrame después de `predict_one()`. Esto deja `last_prediction_` con límites diferentes a los del DataFrame.

Se debe eliminar esa lógica.

Firma recomendada:

```python
def predict(
    self,
    steps: int | None = None,
    recursive: bool = True,
) -> pd.DataFrame:
```

Los parámetros UQ deben quedar fijados en el objeto antes de iniciar la evaluación.

## 5.4. Corregir la actualización recursiva

Aplicar el mismo parámetro:

```python
is_observed: bool = True
```

que en S3-Forecaster.

Nunca llamar a `SequentialACI.update()` con `row["pred"]` como verdad observada.

---

# 6. Cambios en `rolling_evaluation.py`

Archivo: `s3paper/rolling_evaluation.py`

## 6.1. Modificar `rolling_one_step_forecast`

Usar explícitamente:

```python
model.update(float(observation), is_observed=True)
```

Para conservar compatibilidad con modelos que todavía no acepten ese argumento, puede utilizarse temporalmente:

```python
try:
    model.update(float(observation), is_observed=True)
except TypeError:
    model.update(float(observation))
```

Sin embargo, la solución final debe normalizar la interfaz de todos los modelos evaluados.

## 6.2. Validación del intervalo por origen

Antes de guardar cada fila, validar:

```python
lower <= pred <= upper
np.isfinite(pred)
np.isfinite(lower)
np.isfinite(upper)
```

También registrar en el forecast:

```python
alpha_t_before_update
n_conformity_scores_before_update
```

Estos campos permiten demostrar que el intervalo fue construido antes de revelar el target.

## 6.3. Evitar resultados silenciosos cuando el modelo no quedó ajustado

Después de `model.fit(train)`:

```python
if not getattr(model, "fitted", True):
    raise RuntimeError(getattr(model, "fit_report_", "model_not_fitted"))
```

Esto impide producir filas aparentemente válidas después de un `fit()` fallido.

---

# 7. Cambios en `s3_forecaster_experiment.py`

## 7.1. Añadir `seasonal_period` a `optimize_s3_forecaster`

Firma requerida:

```python
def optimize_s3_forecaster(
    train_series,
    *,
    seasonal_period: int = 12,
    n_folds: int = 3,
    validation_size: int = 6,
    n_trials: int = 100,
    seed: int = 42,
    objective_metric: str = "paper_point",
    smape_weight: float = 1.0,
    prior_names=None,
):
```

Eliminar el valor fijo:

```python
seasonal_period=12
```

dentro de `suggest_prior_params()`.

Debe usarse el argumento recibido:

```python
seasonal_period=seasonal_period
```

## 7.2. Sustituir el único holdout por temporal CV

No usar como protocolo principal:

```python
train, validation = temporal_holdout(full, val_size)
```

Usar:

```python
folds = make_expanding_window_folds(
    full,
    n_folds=n_folds,
    validation_size=validation_size,
)
```

Para cada trial:

1. sugerir los parámetros una sola vez;
2. evaluar los mismos parámetros en todos los folds;
3. agregar los scores mediante mediana o media recortada;
4. registrar fallos por fold;
5. devolver infinito si la fracción de fallos excede el umbral permitido.

## 7.3. Crear un sugeridor reutilizable

Extraer la lógica de Optuna a:

```python
def suggest_s3_point_params(
    trial,
    *,
    train_length: int,
    seasonal_period: int,
    prior_names=None,
) -> dict:
```

Esta función debe ser utilizada tanto por HPO uniserie como por HPO multiserie.

No se debe crear un estudio Optuna dentro de otro estudio.

## 7.4. Corregir `optimize_s3_uq`

Los parámetros a optimizar pueden ser:

```python
aci_step_size
interval_scale
min_width_factor
```

El target nominal debe permanecer fijo por defecto:

```python
target_miscoverage = 1.0 - target_coverage
```

Calcular:

```python
minimum_width = min_width_factor * seasonal_naive_scale(
    fold_train,
    seasonal_period=seasonal_period,
)
```

Crear el modelo con:

```python
S3Forecaster(
    ...,
    aci_step_size=aci_step_size,
    target_miscoverage=target_miscoverage,
    interval_scale=interval_scale,
    minimum_width=minimum_width,
)
```

No modificar `forecast["lower"]` ni `forecast["upper"]` después de `evaluate_rolling_model()`.

## 7.5. Corregir `evaluate_s3_forecaster`

Construir el modelo con los parámetros UQ ya integrados:

```python
model = S3Forecaster(
    ...,
    interval_scale=interval_scale,
    minimum_width=minimum_width,
)
```

Eliminar completamente el bloque que reescala límites después del rolling forecast.

Aunque `evaluate_rolling_model()` ya calcule métricas, recalcularlas explícitamente sobre el forecast retornado para impedir métricas obsoletas:

```python
metrics = evaluate_forecast(
    test,
    forecast["pred"],
    y_train=train,
    lower=forecast["lower"],
    upper=forecast["upper"],
    alpha=alpha,
    seasonal_period=seasonal_period,
    elapsed_seconds=elapsed,
    trainable_params=count_trainable_parameters(model),
)
```

## 7.6. Propagar argumentos en `run_s3_experiment`

Añadir y propagar:

```python
seasonal_period
n_folds
validation_size
point_trials
uq_trials
seed
```

El periodo estacional debe ser el mismo en:

- prior HPO;
- temporal CV;
- escalas MASE/RMSSE/MSIS;
- minimum width;
- evaluación final.

---

# 8. Cambios en `s3_fastsketch_experiment.py`

## 8.1. Añadir `seasonal_period` a `evaluate_fastsketch_on_holdout`

Firma:

```python
def evaluate_fastsketch_on_holdout(
    train_subset,
    val_subset,
    params,
    *,
    seasonal_period: int = 12,
    eps: float = 1e-5,
    return_objects: bool = False,
):
```

Eliminar:

```python
seasonal_period=12
```

del constructor interno.

## 8.2. Añadir `seasonal_period` a `optimize_fastsketch_point`

La función debe pasarlo a:

- `suggest_prior_params()`;
- `evaluate_fastsketch_on_holdout()`;
- `evaluate_rolling_model()`;
- `evaluate_forecast()`.

## 8.3. Crear `suggest_fastsketch_point_params`

Extraer las sugerencias a:

```python
def suggest_fastsketch_point_params(
    trial,
    *,
    train_length: int,
    seasonal_period: int,
    prior_names=None,
) -> dict:
```

Esta función debe devolver parámetros serializables. Las tuplas `ema_spans` y `conv_scales` pueden representarse como cadenas durante Optuna y normalizarse únicamente al construir el modelo.

## 8.4. Reemplazar el holdout único por temporal CV

Aplicar la misma política definida para S3-Forecaster.

## 8.5. Corregir `optimize_fastsketch_uq`

El modelo debe recibir directamente:

```python
interval_scale=interval_scale
minimum_width=minimum_width
```

Eliminar el bloque que modifica `forecast["lower"]` y `forecast["upper"]` después del rolling forecast.

## 8.6. Corregir `evaluate_fastsketch`

FastSketch ya recalcula las métricas después del escalado, pero el escalado sigue estando fuera de ACI.

Mover todos los parámetros de intervalo al constructor y eliminar el postprocesamiento.

Después, conservar el recálculo de métricas sobre el forecast final.

## 8.7. Propagar parámetros en `run_fastsketch_experiment`

Añadir los mismos argumentos metodológicos usados por S3-Forecaster para que ambos modelos se comparen bajo un protocolo idéntico.

---

# 9. Reescritura de `multiseries_hpo.py`

Archivo: `s3paper/multiseries_hpo.py`

## 9.1. Problema actual

`optimize_s3_collection()` y `optimize_fastsketch_collection()` crean internamente otro estudio de Optuna con `n_trials=1` usando solo la primera serie.

Esto invalida la relación entre:

```python
trial externo -> parámetros evaluados
```

El estudio externo no controla realmente todas las sugerencias.

## 9.2. Eliminar estudios anidados

Reemplazar:

```python
study = optimize_s3_forecaster(sample, n_trials=1, ...)
return dict(study.trials[0].params)
```

por:

```python
return suggest_s3_point_params(
    trial,
    train_length=minimum_fold_train_length,
    seasonal_period=seasonal_period,
    prior_names=prior_names,
)
```

Aplicar lo mismo para FastSketch.

## 9.3. Precalcular los folds antes de iniciar Optuna

Dentro de `optimize_collection`, construir una estructura fija:

```python
fold_map = {
    series_id: make_expanding_window_folds(...)
    for series_id, series in sorted(series_map.items())
}
```

Ventajas:

- todos los trials usan exactamente los mismos folds;
- se detectan series sin folds válidos antes de iniciar HPO;
- se conoce el mínimo tamaño de entrenamiento para definir espacios de búsqueda seguros;
- el orden es determinista.

## 9.4. Cambiar la firma del `objective_factory`

Usar:

```python
objective_factory(
    train,
    validation,
    params,
    *,
    series_id,
    fold_id,
)
```

Esto facilita cache, auditoría y registro de fallos.

## 9.5. Agregación primaria

Para el objetivo point recomendado:

```python
score = MASE + lambda_smape * sMAPE_fraction
```

Agregar primero por serie y luego entre series para evitar que una serie con más folds domine el objetivo:

```text
score por fold
→ mediana por serie
→ mediana entre series
```

No agregar todos los folds directamente como si fueran observaciones intercambiables.

## 9.6. Control de fallos

Por trial registrar:

```python
n_series
n_folds_total
n_successes
n_failures
failure_fraction
failed_series
failed_folds
```

Reglas:

- cero evaluaciones exitosas: `inf`;
- fracción de fallos superior al umbral: `inf`;
- prior obligatorio que falla sistemáticamente: trial inválido;
- no sustituir silenciosamente el modelo por Naive.

## 9.7. Implementar HPO UQ multiserie

Añadir:

```python
optimize_s3_uq_collection(...)
optimize_fastsketch_uq_collection(...)
```

Entradas:

```python
series_map
frozen_point_params
target_coverage
seasonal_period
n_trials
n_folds
validation_size
```

Espacio UQ:

```python
aci_step_size
interval_scale
min_width_factor
```

El objetivo por fold debe usar:

```python
MSIS + coverage_penalty
```

con target nominal fijo.

Los parámetros point no deben volver a optimizarse durante UQ HPO.

---

# 10. Ajustes en `ConformalizedBaseline`

Archivo: `s3paper/baselines.py`

## 10.1. Usar el helper común de intervalos

En `predict_one()`, reemplazar el cálculo local por `finalize_symmetric_interval()`.

Esto garantiza que todos los modelos usan la misma definición de:

- `interval_scale`;
- `minimum_width`;
- semi-anchura;
- intervalo final entregado a ACI.

## 10.2. Normalizar la interfaz `update`

Modificar a:

```python
def update(self, observation: float, *, is_observed: bool = True):
```

ACI solo se actualiza cuando `is_observed=True`.

## 10.3. Separar construcción del modelo y transformación

Añadir parámetros opcionales:

```python
transformer_factory=None
model_factory=None
```

Durante cada ajuste causal:

1. construir un transformer nuevo;
2. ajustarlo solo con el historial disponible;
3. transformar el historial;
4. ajustar el baseline;
5. invertir la predicción.

Esto permite evaluar ARIMA u otros modelos con log-transformación positiva sin incorporar información futura.

## 10.4. Limpiar la actualización del historial

Reemplazar:

```python
history.loc[_.__class__(_) if False else _] = float(observed)
```

por:

```python
history.loc[timestamp] = float(observed)
```

El nombre de variable debe ser explícito.

---

# 11. Integración del cache de priors

Archivo: `s3paper/prior_cache.py`

## 11.1. Ampliar la clave

La clave actual usa nombre, parámetros, historial y horizonte. Añadir:

```python
dataset_id
series_id
fold_id
forecast_origin
seasonal_period
code_version_or_commit
```

Firma sugerida:

```python
def make_key(
    self,
    *,
    dataset_id: str,
    series_id: str,
    fold_id: str,
    forecast_origin,
    prior_name: str,
    prior_params: dict | None,
    history_values,
    seasonal_period: int,
    horizon: int = 1,
    code_version: str = "unknown",
) -> str:
```

## 11.2. Cachear trayectorias causales, no objetos mutables

Guardar datos inmutables:

```python
{
    "fitted_values": ...,
    "forecast_values": ...,
    "forecast_index": ...,
    "status": ...,
}
```

No guardar una instancia de prior ya mutada.

## 11.3. Integración recomendada

Añadir un helper:

```python
def compute_causal_prior_path(
    prior_name,
    prior_params,
    train,
    validation,
    *,
    cache=None,
    cache_context=None,
):
```

Debe producir exactamente las predicciones one-step que usaría el prior durante el recorrido causal.

El cache debe ser opcional y no cambiar resultados numéricos.

---

# 12. Consolidación de métricas

Archivo: `s3paper/metrics.py`

## 12.1. Fuente única

Toda evaluación final debe pasar por:

```python
evaluate_forecast(
    y_true,
    y_pred,
    y_train=train,
    lower=lower,
    upper=upper,
    seasonal_period=seasonal_period,
    alpha=alpha,
)
```

No deben existir cálculos locales alternativos de:

- MAPE;
- sMAPE;
- MASE;
- RMSSE;
- ECP;
- MIS;
- MSIS.

## 12.2. Validaciones

Añadir comprobaciones:

```python
len(y_true) == len(y_pred)
len(lower) == len(y_true)
len(upper) == len(y_true)
lower <= upper
```

Cuando se soliciten MASE, RMSSE o MSIS, exigir:

- `y_train`, o
- escala precomputada explícita.

Nunca inferir la escala a partir del test.

---

# 13. Pruebas requeridas antes de considerar el paquete listo

## 13.1. Intervalo online

### `test_s3_update_uses_reported_interval`

1. ejecutar `predict_one()`;
2. guardar `lower` y `upper`;
3. llamar `update(real_target)`;
4. comprobar que el miss de ACI se calculó con esos mismos límites.

### `test_fastsketch_update_uses_reported_interval`

Misma prueba para FastSketch.

### `test_interval_scaling_occurs_before_target_reveal`

Cambiar `interval_scale` y verificar que el miss y `alpha_t` cambian de acuerdo con el intervalo escalado.

## 13.2. Métricas consistentes

### `test_s3_returned_metrics_match_returned_forecast`

Recalcular ECP, MIS y MSIS desde el DataFrame retornado y exigir igualdad numérica con `result["metrics"]`.

### `test_fastsketch_returned_metrics_match_returned_forecast`

Misma prueba para FastSketch.

## 13.3. Seasonal period

### `test_seasonal_period_is_propagated`

Usar un valor no estándar, por ejemplo `4`, y comprobar que llega a:

- prior;
- escalas;
- minimum width;
- evaluación;
- HPO.

## 13.4. HPO multiserie

### `test_multiseries_hpo_has_no_nested_study`

Mockear `optuna.create_study` y comprobar que solo se crea un estudio.

### `test_same_params_are_used_across_all_series_and_folds`

Registrar los parámetros recibidos por `objective_factory` y verificar igualdad dentro del mismo trial.

### `test_trial_params_equal_evaluated_params`

Comprobar que `study.best_params` coincide exactamente con la configuración evaluada.

### `test_hpo_does_not_access_evaluation_series`

Perturbar series no incluidas en `series_map` y verificar que `best_params` no cambian.

## 13.5. UQ HPO

### `test_uq_hpo_keeps_point_params_frozen`

Comprobar que ningún parámetro point aparece como sugerencia durante UQ HPO.

### `test_nominal_coverage_fixed_by_default`

Verificar que `target_miscoverage` no forma parte de `best_params` salvo que `optimize_nominal_level=True`.

## 13.6. Predicción recursiva

### `test_recursive_prediction_does_not_update_aci`

Guardar:

```python
alpha_t
len(conformity_scores)
```

antes y después de `predict(steps > 1)`.

Deben permanecer sin cambios.

## 13.7. Baselines

### `test_conformalized_baseline_uses_final_interval_for_update`

La misma condición de identidad intervalo-reportado/intervalo-actualizado.

### `test_transformed_arima_round_trip`

Comprobar transformación, predicción e inversión sin valores no finitos.

## 13.8. Cache

### `test_prior_cache_does_not_change_predictions`

Las predicciones con y sin cache deben ser idénticas.

### `test_prior_cache_key_changes_with_fold_or_origin`

La clave debe cambiar si cambia cualquiera de los elementos causales relevantes.

---

# 14. Orden recomendado de modificación

## Fase 1: consistencia de intervalos

1. `conformal.py`: helper de intervalo final.
2. `s3_forecaster.py`: parámetros UQ dentro del modelo.
3. `s3_fastsketch.py`: parámetros UQ dentro del modelo.
4. eliminar postprocesamiento en ambos `*_experiment.py`.
5. corregir métricas de S3-Forecaster.
6. corregir predicción recursiva sin actualización artificial de ACI.

## Fase 2: protocolo de HPO

7. extraer `suggest_s3_point_params()`.
8. extraer `suggest_fastsketch_point_params()`.
9. propagar `seasonal_period`.
10. reemplazar holdout único por expanding-window CV.
11. reescribir HPO multiserie sin estudios anidados.
12. añadir HPO UQ multiserie.

## Fase 3: comparabilidad

13. normalizar `ConformalizedBaseline`.
14. integrar transformaciones causales.
15. consolidar métricas.
16. integrar cache de priors.

## Fase 4: validación

17. ejecutar las pruebas unitarias nuevas.
18. ejecutar `compileall`.
19. ejecutar toda la suite de `pytest`.
20. confirmar que no hay diferencias entre métricas recalculadas y métricas retornadas.

---

# 15. Definición de terminado

El paquete puede considerarse funcionalmente listo para regenerar resultados cuando se cumplan todas estas condiciones:

- el intervalo reportado es exactamente el intervalo usado por ACI para actualizar `alpha_t`;
- no existe escalado de intervalos después del rolling forecast;
- S3-Forecaster y FastSketch recalculan métricas desde el forecast final;
- `seasonal_period` no está fijado silenciosamente en `12` dentro del HPO;
- el HPO point usa múltiples folds temporales;
- el HPO multiserie usa un único `trial` real y una configuración compartida;
- el HPO UQ mantiene congelados los parámetros point;
- el target nominal de cobertura permanece fijo por defecto;
- las predicciones recursivas no actualizan ACI con pseudobjetivos;
- baselines y modelos propuestos comparten la misma lógica de intervalos;
- las escalas de MASE, RMSSE y MSIS provienen únicamente del entrenamiento;
- todas las pruebas de consistencia pasan;
- no hay fallos silenciosos ni sustituciones automáticas no reportadas.

---

# 16. Veredicto técnico

Después de excluir notebooks, Kaggle, loaders y generación material de resultados, los bloqueadores restantes se concentran en cuatro núcleos:

1. **consistencia online de los intervalos**;
2. **consistencia entre forecast y métricas**;
3. **HPO temporal y multiserie correctamente implementado**;
4. **propagación uniforme de parámetros metodológicos**.

Los cambios descritos en este documento son suficientes para que el paquete quede metodológicamente preparado para una posterior regeneración de resultados. La prioridad absoluta debe ser eliminar el escalado post-hoc de intervalos y corregir el HPO multiserie anidado, porque ambos problemas pueden modificar directamente las conclusiones cuantitativas del artículo.
