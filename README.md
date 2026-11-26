# Prototipo de simulación — Cribado de cáncer de próstata a 5 años (CRISP-DM)

Prueba de concepto en Python asociada al entregable CRISP-DM de cribado poblacional de cáncer de próstata a 5 años (hombres a los 45 años). Valida el flujo de modelado, calibración de probabilidades y selección de umbral $\tau$ bajo la restricción de seguridad oncológica: **límite superior exacto Clopper–Pearson del IC 95 % de la tasa de falsos negativos (FNR) < 1 %**.

> **Nota metodológica:** Los datos se generan de forma sintética (`n = 5000`, semilla `42`) para probar el *pipeline* y analizar comportamientos de los modelos; **no** corresponden a la cohorte histórica real (`2006–2021`) ni constituyen un sistema clínico listo para producción.

## Qué valida este prototipo

* **Partición secuencial (`60 % / 15 % / 25 %`):** separa entrenamiento, validación y prueba sin barajar para replicar la lógica de corte temporal del entregable.
* **Modelos comparados:** regresión logística con *splines* cúbicos en `psa_total` frente a `RandomForestClassifier` e `HistGradientBoostingClassifier` con pesos de clase por desbalance.
* **Calibración y umbral:** compara calibración de Platt (`sigmoid`) e isotónica. La FNR se calcula entre los casos positivos reales; su límite superior se estima con el intervalo binomial exacto Clopper–Pearson bilateral del 95 % (extremo superior al 97,5 %). Se selecciona en validación el umbral más alto cuyo límite superior sea inferior a `1 %`, y se contrasta frente a `PSA >= 3.0` en prueba.
* **Alcance de la restricción:** con pocos casos positivos en validación, el límite exacto puede ser mucho mayor que `1 %` incluso si no se observan falsos negativos; en ese caso el prototipo informa que no hay evidencia suficiente para seleccionar un umbral seguro.

## Ejecución

```bash
pip install -r requirements.txt
python simulacion_cribado_prostata.py