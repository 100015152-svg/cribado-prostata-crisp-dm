"""
Prototipo de simulación y evaluación de umbral para cribado de cáncer de próstata a 5 años
(cohorte sintética de ~45 años, partición secuencial train / val / test = 60% / 15% / 25%).

Nota metodológica:
- Es una simulación sintética para validar el pipeline (splines, boosting ponderado,
  calibración Platt/isotónica y selección de umbral por límite superior del IC 95% de FN < 1%),
  no la serie histórica real (2006-2021) ni un sistema listo para producción.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def simular_cohorte(n: int = 5000, seed: int = 42) -> pd.DataFrame:
  rng = np.random.default_rng(seed)

  edad_exacta = rng.uniform(44.95, 45.35, size=n)
  antecedentes = rng.binomial(1, 0.14, size=n)
  tacto_rectal = rng.choice(
      ["normal", "sospechoso", "no_realizado"], size=n, p=[0.82, 0.10, 0.08]
  )
  imc = np.clip(rng.normal(26.5, 3.8, size=n), 17.0, 42.0)
  psa_total = np.clip(rng.lognormal(mean=0.35, sigma=0.65, size=n), 0.15, 12.0)
  ratio_psa = np.clip(rng.beta(a=4.5, b=14.0, size=n), 0.04, 0.65)

  tacto_ser = pd.Series(tacto_rectal, dtype="object")
  tacto_ser[rng.random(size=n) < 0.03] = np.nan

  imc_ser = pd.Series(imc)
  imc_ser[rng.random(size=n) < 0.04] = np.nan

  efecto_tacto = np.where(tacto_rectal == "sospechoso", 1.15, 0.0)
  efecto_ratio = -1.6 * np.clip(ratio_psa - 0.25, -0.20, 0.25)

  log_odds_real = (
      -4.7
      + 0.45 * np.maximum(psa_total - 2.0, 0) ** 1.35
      + 0.35 * np.sin(psa_total * 1.1)
      + 0.85 * antecedentes
      + efecto_tacto
      + efecto_ratio
      + 0.4 * (edad_exacta - 45.0)
  )
  prob_real = 1.0 / (1.0 + np.exp(-log_odds_real))
  cancer_5a = rng.binomial(1, prob_real)

  return pd.DataFrame({
      "edad_exacta": edad_exacta,
      "psa_total": psa_total,
      "ratio_psa": ratio_psa,
      "imc": imc_ser,
      "antecedentes_familiares": antecedentes,
      "tacto_rectal": tacto_ser,
      "cancer_5a": cancer_5a,
  })


def construir_preprocesador(
    num_spline: list[str], num_lineal: list[str], cat_cols: list[str]
) -> ColumnTransformer:
  pipe_psa_spline = Pipeline([
      ("imp", SimpleImputer(strategy="median")),
      (
          "spline",
          SplineTransformer(
              n_knots=4, degree=3, knots="quantile", include_bias=False
          ),
      ),
  ])
  pipe_num_lineal = Pipeline([
      ("imp", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
  ])
  pipe_cat = Pipeline([
      ("imp", SimpleImputer(strategy="most_frequent")),
      ("ohe", OneHotEncoder(handle_unknown="ignore")),
  ])

  return ColumnTransformer([
      ("psa_sp", pipe_psa_spline, num_spline),
      ("num_lin", pipe_num_lineal, num_lineal),
      ("cat", pipe_cat, cat_cols),
  ])


def evaluar_umbral(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    n_boot: int = 500,
) -> dict[str, float]:
  pred_pos = (probs >= tau).astype(int)
  fn_mask = ((y_true == 1) & (pred_pos == 0)).astype(float)
  fn_pct = fn_mask.mean() * 100.0

  boot_fn = [
      fn_mask[rng.integers(0, len(fn_mask), size=len(fn_mask))].mean() * 100.0
      for _ in range(n_boot)
  ]
  ci95_sup = float(np.percentile(boot_fn, 97.5))
  fp_pct = float(((y_true == 0) & (pred_pos == 1)).mean() * 100.0)
  deriv_pct = float(pred_pos.mean() * 100.0)

  return {
      "umbral": float(tau),
      "fn_pct": float(fn_pct),
      "ci95_sup_pct": ci95_sup,
      "fp_pct": fp_pct,
      "deriv_pct": deriv_pct,
  }


def generar_graficas(
    df_umb: pd.DataFrame,
    train_psa: pd.Series,
    modelo_spline: Pipeline,
    hgb_base: Pipeline,
    hgb_platt: CalibratedClassifierCV,
    hgb_iso: CalibratedClassifierCV,
    umbral_elegido: float | None,
    output_path: Path,
) -> None:
  psa_grid = np.linspace(0.2, 10.0, 300)
  X_grid = pd.DataFrame({
      "edad_exacta": np.full_like(psa_grid, 45.1),
      "psa_total": psa_grid,
      "ratio_psa": np.full_like(psa_grid, 0.22),
      "imc": np.full_like(psa_grid, 26.5),
      "antecedentes_familiares": np.zeros_like(psa_grid, dtype=int),
      "tacto_rectal": np.full(len(psa_grid), "normal", dtype=object),
  })

  p_grid_real = 1.0 / (
      1.0
      + np.exp(
          -(
              -4.7
              + 0.45 * np.maximum(psa_grid - 2.0, 0) ** 1.35
              + 0.35 * np.sin(psa_grid * 1.1)
              - 1.6 * np.clip(0.22 - 0.25, -0.20, 0.25)
              + 0.4 * (45.1 - 45.0)
          )
      )
  )
  nudos_psa = np.quantile(train_psa, np.linspace(0, 1, 4))

  fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

  axes[0, 0].plot(psa_grid, p_grid_real, label="Real (simulada)", lw=2.2)
  axes[0, 0].plot(
      psa_grid,
      modelo_spline.predict_proba(X_grid)[:, 1],
      label="Logística + splines",
      lw=2.0,
  )
  for k in nudos_psa:
    axes[0, 0].axvline(k, color="gray", ls=":", lw=1.0)
  axes[0, 0].set_title("1. Curva con splines y nudos")
  axes[0, 0].set_xlabel("PSA total (ng/mL)")
  axes[0, 0].set_ylabel("Probabilidad a 5 años")
  axes[0, 0].legend(fontsize=8)
  axes[0, 0].grid(alpha=0.25)

  axes[0, 1].plot(psa_grid, p_grid_real, label="Real (simulada)", lw=2.2)
  axes[0, 1].plot(
      psa_grid,
      hgb_base.predict_proba(X_grid)[:, 1],
      label="HGB pesos (sin calibrar)",
      ls="--",
      lw=1.6,
  )
  axes[0, 1].plot(
      psa_grid,
      hgb_platt.predict_proba(X_grid)[:, 1],
      label="HGB calibrado Platt",
      lw=2.0,
  )
  axes[0, 1].set_title("2. Pesos de clase y calibración (HGB)")
  axes[0, 1].set_xlabel("PSA total (ng/mL)")
  axes[0, 1].set_ylabel("Probabilidad a 5 años")
  axes[0, 1].legend(fontsize=8)
  axes[0, 1].grid(alpha=0.25)

  axes[1, 0].plot(psa_grid, p_grid_real, label="Real (simulada)", lw=2.2)
  axes[1, 0].plot(
      psa_grid,
      hgb_platt.predict_proba(X_grid)[:, 1],
      label="HGB Platt (sigmoid)",
      lw=2.0,
  )
  axes[1, 0].plot(
      psa_grid,
      hgb_iso.predict_proba(X_grid)[:, 1],
      label="HGB Isotónica",
      ls="-.",
      lw=2.0,
  )
  axes[1, 0].set_title("3. Calibración Platt vs. Isotónica")
  axes[1, 0].set_xlabel("PSA total (ng/mL)")
  axes[1, 0].set_ylabel("Probabilidad a 5 años")
  axes[1, 0].legend(fontsize=8)
  axes[1, 0].grid(alpha=0.25)

  axes[1, 1].plot(df_umb["umbral"], df_umb["fn_pct"], label="% FN (val)", lw=2.0)
  axes[1, 1].plot(
      df_umb["umbral"],
      df_umb["ci95_sup_pct"],
      label="Límite sup. IC 95% FN (val)",
      ls="--",
      lw=2.0,
  )
  axes[1, 1].plot(df_umb["umbral"], df_umb["fp_pct"], label="% FP (val)", lw=1.8)
  axes[1, 1].plot(
      df_umb["umbral"], df_umb["deriv_pct"], label="% Derivaciones (val)", ls=":", lw=1.8
  )
  axes[1, 1].axhline(
      1.0, color="black", ls="-.", lw=1.2, label="Techo seguridad FN (1%)"
  )
  if umbral_elegido is not None:
    axes[1, 1].axvline(
        umbral_elegido,
        color="green",
        lw=2.0,
        label=f"Umbral ({umbral_elegido:.3f})",
    )
  axes[1, 1].set_title("4. Selección de umbral seguro (en VAL)")
  axes[1, 1].set_xlabel("Umbral de probabilidad (τ)")
  axes[1, 1].set_ylabel("Porcentaje sobre cohorte val (%)")
  axes[1, 1].legend(fontsize=7.5, loc="upper right")
  axes[1, 1].grid(alpha=0.25)

  plt.tight_layout()
  plt.savefig(output_path, dpi=200)
  plt.close(fig)


def main() -> None:
  n = 5000
  seed = 42
  rng = np.random.default_rng(seed)
  output_dir = Path("outputs")
  output_dir.mkdir(parents=True, exist_ok=True)

  df = simular_cohorte(n=n, seed=seed)

  # Partición secuencial sin barajar (60% train / 15% val / 25% test)
  train_end = int(0.60 * n)
  val_end = int(0.75 * n)

  train = df.iloc[:train_end].copy()
  val = df.iloc[train_end:val_end].copy()
  test = df.iloc[val_end:].copy()

  num_spline = ["psa_total"]
  num_lineal = ["edad_exacta", "ratio_psa", "imc"]
  cat_cols = ["antecedentes_familiares", "tacto_rectal"]
  features = num_spline + num_lineal + cat_cols

  X_train, y_train = train[features], train["cancer_5a"]
  X_val, y_val = val[features], val["cancer_5a"]
  X_test, y_test = test[features], test["cancer_5a"]

  prep = construir_preprocesador(num_spline, num_lineal, cat_cols)

  modelo_spline = Pipeline(
      [("prep", prep), ("clf", LogisticRegression(max_iter=1000))]
  )

  n_neg = (y_train == 0).sum()
  n_pos = max((y_train == 1).sum(), 1)
  pesos_train = np.where(y_train == 1, n_neg / n_pos, 1.0)

  rf_base = Pipeline([
      ("prep", prep),
      (
          "clf",
          RandomForestClassifier(
              n_estimators=300,
              max_depth=5,
              min_samples_leaf=25,
              class_weight="balanced",
              random_state=seed,
          ),
      ),
  ])

  hgb_base = Pipeline([
      ("prep", prep),
      (
          "clf",
          HistGradientBoostingClassifier(
              max_iter=150,
              max_depth=3,
              learning_rate=0.05,
              min_samples_leaf=25,
              random_state=seed,
          ),
      ),
  ])

  modelo_spline.fit(X_train, y_train)
  rf_base.fit(X_train, y_train)
  hgb_base.fit(X_train, y_train, clf__sample_weight=pesos_train)

  hgb_platt = CalibratedClassifierCV(hgb_base, method="sigmoid", cv=5)
  hgb_platt.fit(X_train, y_train)

  hgb_iso = CalibratedClassifierCV(hgb_base, method="isotonic", cv=5)
  hgb_iso.fit(X_train, y_train)

  modelos = {
      "Logística + splines": modelo_spline,
      "RF pesos (sin calibrar)": rf_base,
      "HGB pesos (sin calibrar)": hgb_base,
      "HGB calibrado Platt": hgb_platt,
      "HGB calibrado Isotónica": hgb_iso,
  }

  preds_val = {
      nombre: m.predict_proba(X_val)[:, 1] for nombre, m in modelos.items()
  }
  preds_test = {
      nombre: m.predict_proba(X_test)[:, 1] for nombre, m in modelos.items()
  }

  print("\n--- Métricas en VAL ---")
  for nombre, p in preds_val.items():
    print(
        f"{nombre:<26} | PR-AUC: {average_precision_score(y_val, p):.4f} "
        f"| Brier: {brier_score_loss(y_val, p):.4f}"
    )

  print("\n--- Métricas en TEST ---")
  for nombre, p in preds_test.items():
    print(
        f"{nombre:<26} | PR-AUC: {average_precision_score(y_test, p):.4f} "
        f"| Brier: {brier_score_loss(y_test, p):.4f}"
    )

  umbrales = np.linspace(0.005, 0.15, 80)
  filas_val = [
      evaluar_umbral(y_val.values, preds_val["HGB calibrado Platt"], tau, rng)
      for tau in umbrales
  ]
  df_umb = pd.DataFrame(filas_val)

  validos = df_umb[df_umb["ci95_sup_pct"] < 1.0]
  umbral_elegido = float(validos["umbral"].max()) if not validos.empty else None

  base_pos_test = (test["psa_total"].values >= 3.0).astype(int)
  base_fp = float(((y_test.values == 0) & (base_pos_test == 1)).mean() * 100.0)
  base_fn = float(((y_test.values == 1) & (base_pos_test == 0)).mean() * 100.0)
  base_deriv = float(base_pos_test.mean() * 100.0)

  print(
      f"\nProtocolo base en TEST (PSA >= 3.0) -> Derivaciones: {base_deriv:.2f}% | "
      f"FP: {base_fp:.2f}% | FN: {base_fn:.2f}%"
  )

  if umbral_elegido is not None:
    res_test = evaluar_umbral(
        y_test.values, preds_test["HGB calibrado Platt"], umbral_elegido, rng
    )
    red_fp = ((base_fp - res_test["fp_pct"]) / max(base_fp, 1e-9)) * 100.0
    red_deriv = (
        (base_deriv - res_test["deriv_pct"]) / max(base_deriv, 1e-9)
    ) * 100.0
    print(f"Umbral elegido (seleccionado en VAL): {umbral_elegido:.4f}")
    print(
        f"En TEST con umbral elegido -> Derivaciones: {res_test['deriv_pct']:.2f}% | "
        f"FP: {res_test['fp_pct']:.2f}% | FN: {res_test['fn_pct']:.2f}% "
        f"(IC 95% sup: {res_test['ci95_sup_pct']:.2f}%)"
    )
    print(
        f"Reducción relativa en TEST vs. protocolo base -> Derivaciones: {red_deriv:.1f}% | "
        f"FP: {red_fp:.1f}%"
    )
  else:
    print(
        "Ningún umbral en el rango evaluado cumple IC 95% superior de FN < 1% en VAL."
    )

  fig_path = output_dir / "todas_las_graficas_prostata.png"
  generar_graficas(
      df_umb=df_umb,
      train_psa=train["psa_total"],
      modelo_spline=modelo_spline,
      hgb_base=hgb_base,
      hgb_platt=hgb_platt,
      hgb_iso=hgb_iso,
      umbral_elegido=umbral_elegido,
      output_path=fig_path,
  )
  print(f"\nGuardada figura en: {fig_path}")


if __name__ == "__main__":
  main()