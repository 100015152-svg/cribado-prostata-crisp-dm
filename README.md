# Prototipo de simulación — Cribado de cáncer de próstata a 5 años (CRISP-DM)

Prueba de concepto en Python asociada al entregable CRISP-DM de cribado poblacional de cáncer de próstata a 5 años (hombres a los 45 años). Valida el flujo de modelado, calibración de probabilidades y selección de umbral $\tau$ bajo la restricción de seguridad oncológica: **límite superior del IC 95 % de falsos negativos < 1 %**.

> **Nota metodológica:** Los datos se generan de forma sintética (`n = 5000`, semilla `42`) para probar el *pipeline* y analizar comportamientos de los modelos; **no** corresponden a la cohorte histórica real (`2006–2021`) ni constituyen un sistema clínico listo para producción.

## Qué valida este prototipo

* **Partición secuencial (`60 % / 15 % / 25 %`):** separa entrenamiento, validación y prueba sin barajar para replicar la lógica de corte temporal del entregable.
* **Modelos comparados:** regresión logística con *splines* cúbicos en `psa_total` frente a `RandomForestClassifier` e `HistGradientBoostingClassifier` con pesos de clase por desbalance.
* **Calibración y umbral:** compara calibración de Platt (`sigmoid`) e isotónica, selecciona en validación el umbral $\tau$ más alto cuyo límite superior del IC 95 % de falsos negativos (vía *bootstrap*) quede por debajo del `1 %`, y contrasta frente a un corte basal `PSA >= 3.0` en prueba.

## Ejecución

```bash
pip install -r requirements.txt
python simulacion_cribado_prostata.py