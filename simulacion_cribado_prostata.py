"""
Prototipo de simulación y evaluación de umbral para cribado de cáncer de próstata a 5 años
(cohorte sintética de ~45 años, partición secuencial train / val / test = 60% / 15% / 25%).

Nota metodológica:
- Es una simulación sintética para validar el pipeline (splines, boosting ponderado,
  calibración Platt/isotónica, comparación bootstrap vs. Clopper-Pearson y selección
  de umbral con límite superior exacto Clopper-Pearson del IC 95% de la FNR < 1%),
  no la serie histórica real (2006-2021) ni un sistema listo para producción.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta
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
    rng: np.random.Generator | None = None,
    n_boot: int = 500,
) -> dict[str, float]:
  pred_pos = (probs >= tau).astype(int)
  positivos = y_true == 1
  n_positivos = int(positivos.sum())
  if n_positivos == 0:
    raise ValueError("No se puede estimar la FNR sin casos positivos reales.")

  fn_en_positivos = ((pred_pos == 0)[positivos]).astype(float)
  n_fn = int(fn_en_positivos.sum())
  fn_pct = n_fn / n_positivos * 100.0

  # Límite superior exacto bilateral del IC 95% de Clopper-Pearson sobre FNR
  ci95_sup = (
      100.0
      if n_fn == n_positivos
      else float(beta.ppf(0.975, n_fn + 1, n_positivos - n_fn) * 100.0)
  )

  # Límite superior del IC 95% por bootstrap (remuestreo con reemplazo sobre positivos)
  if rng is not None and n_boot > 0:
    idx_boot = rng.integers(0, n_positivos, size=(n_boot, n_positivos))
    boot_fn = fn_en_positivos[idx_boot].mean(axis=1) * 100.0
    boot_ci95_sup = float(np.percentile(boot_fn, 97.5))
  else:
    boot_ci95_sup = np.nan

  fp_pct = float(((y_true == 0) & (pred_pos == 1)).mean() * 100.0)
  deriv_pct = float(pred_pos.mean() * 100.0)

  return {
      "umbral": float(tau),
      "fn_pct": float(fn_pct),
      "ci95_sup_pct": ci95_sup,
      "boot_ci95_sup_pct": boot_ci95_sup,
      "fp_pct": fp_pct,
      "deriv_pct": deriv_pct,
  }


def construir_malla_psa(train_psa: pd.Series):
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
  return psa_grid, X_grid, p_grid_real, nudos_psa


def guardar_png_seguro(fig: plt.Figure, output_path: Path, fallback_dir: Path) -> Path:
  try:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
  except OSError:
    fallback_path = (fallback_dir / output_path.name).resolve()
    fig.savefig(fallback_path.as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fallback_path


def guardar_texto_seguro(contenido: str, output_path: Path, fallback_dir: Path) -> Path:
  try:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contenido, encoding="utf-8")
    return output_path
  except OSError:
    fallback_path = (fallback_dir / output_path.name).resolve()
    fallback_path.write_text(contenido, encoding="utf-8")
    return fallback_path


def generar_graficas_png(
    df_umb: pd.DataFrame,
    train_psa: pd.Series,
    modelo_spline: Pipeline,
    hgb_base: Pipeline,
    hgb_platt: CalibratedClassifierCV,
    hgb_iso: CalibratedClassifierCV,
    umbral_elegido: float | None,
    output_path: Path,
    fallback_dir: Path,
) -> Path:
  psa_grid, X_grid, p_grid_real, nudos_psa = construir_malla_psa(train_psa)

  fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.0))

  # 1. Curva con splines y nudos
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

  # 2. Pesos de clase y calibración (HGB)
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

  # 3. Calibración Platt vs. Isotónica
  axes[0, 2].plot(psa_grid, p_grid_real, label="Real (simulada)", lw=2.2)
  axes[0, 2].plot(
      psa_grid,
      hgb_platt.predict_proba(X_grid)[:, 1],
      label="HGB Platt (sigmoid)",
      lw=2.0,
  )
  axes[0, 2].plot(
      psa_grid,
      hgb_iso.predict_proba(X_grid)[:, 1],
      label="HGB Isotónica",
      ls="-.",
      lw=2.0,
  )
  axes[0, 2].set_title("3. Calibración Platt vs. Isotónica")
  axes[0, 2].set_xlabel("PSA total (ng/mL)")
  axes[0, 2].set_ylabel("Probabilidad a 5 años")
  axes[0, 2].legend(fontsize=8)
  axes[0, 2].grid(alpha=0.25)

  # 4. Selección de umbral con Clopper-Pearson
  axes[1, 0].plot(
      df_umb["umbral"],
      df_umb["fn_pct"],
      label="FNR: FN entre positivos (val)",
      lw=2.0,
  )
  axes[1, 0].plot(
      df_umb["umbral"],
      df_umb["ci95_sup_pct"],
      label="Límite sup. exacto Clopper-Pearson 95% (val)",
      ls="--",
      lw=2.0,
  )
  axes[1, 0].plot(df_umb["umbral"], df_umb["fp_pct"], label="% FP (val)", lw=1.8)
  axes[1, 0].plot(
      df_umb["umbral"],
      df_umb["deriv_pct"],
      label="% Derivaciones (val)",
      ls=":",
      lw=1.8,
  )
  axes[1, 0].axhline(
      1.0, color="black", ls="-.", lw=1.2, label="Techo seguridad FN (1%)"
  )
  if umbral_elegido is not None:
    axes[1, 0].axvline(
        umbral_elegido,
        color="green",
        lw=2.0,
        label=f"Umbral ({umbral_elegido:.3f})",
    )
  axes[1, 0].set_title("4. Selección de umbral con Clopper-Pearson (VAL)")
  axes[1, 0].set_xlabel("Umbral de probabilidad (τ)")
  axes[1, 0].set_ylabel(
      "Porcentaje (%; FNR sobre positivos, FP/derivaciones sobre cohorte)"
  )
  axes[1, 0].legend(fontsize=7.5, loc="upper right")
  axes[1, 0].grid(alpha=0.25)

  # 5. Bootstrap en FNR (VAL)
  axes[1, 1].plot(
      df_umb["umbral"],
      df_umb["fn_pct"],
      label="FNR observada (val)",
      lw=2.0,
  )
  axes[1, 1].plot(
      df_umb["umbral"],
      df_umb["boot_ci95_sup_pct"],
      label="Límite sup. Bootstrap 95% (val)",
      color="purple",
      ls="--",
      lw=2.0,
  )
  axes[1, 1].axhline(
      1.0, color="black", ls="-.", lw=1.2, label="Techo seguridad FN (1%)"
  )
  axes[1, 1].set_title("5. Estimación por Bootstrap en FNR (VAL)")
  axes[1, 1].set_xlabel("Umbral de probabilidad (τ)")
  axes[1, 1].set_ylabel("FNR / Límite superior (%)")
  axes[1, 1].legend(fontsize=8)
  axes[1, 1].grid(alpha=0.25)

  # 6. Comparativa Bootstrap vs. Clopper-Pearson
  axes[1, 2].plot(
      df_umb["umbral"],
      df_umb["fn_pct"],
      label="FNR observada (val)",
      lw=2.0,
  )
  axes[1, 2].plot(
      df_umb["umbral"],
      df_umb["boot_ci95_sup_pct"],
      label="Bootstrap IC 95% sup.",
      color="purple",
      ls=":",
      lw=2.0,
  )
  axes[1, 2].plot(
      df_umb["umbral"],
      df_umb["ci95_sup_pct"],
      label="Clopper-Pearson IC 95% sup.",
      color="darkorange",
      ls="--",
      lw=2.0,
  )
  axes[1, 2].axhline(
      1.0, color="black", ls="-.", lw=1.2, label="Techo seguridad FN (1%)"
  )
  axes[1, 2].set_title("6. Bootstrap vs. Clopper-Pearson (VAL)")
  axes[1, 2].set_xlabel("Umbral de probabilidad (τ)")
  axes[1, 2].set_ylabel("FNR / Límite superior (%)")
  axes[1, 2].legend(fontsize=8)
  axes[1, 2].grid(alpha=0.25)

  return guardar_png_seguro(fig, output_path, fallback_dir)


def generar_graficas_html(
    df_umb: pd.DataFrame,
    train_psa: pd.Series,
    modelo_spline: Pipeline,
    hgb_base: Pipeline,
    hgb_platt: CalibratedClassifierCV,
    hgb_iso: CalibratedClassifierCV,
    umbral_elegido: float | None,
    output_html: Path,
    fallback_dir: Path,
) -> Path:
  psa_grid, X_grid, p_grid_real, nudos_psa = construir_malla_psa(train_psa)

  fig = make_subplots(
      rows=2,
      cols=3,
      subplot_titles=(
          "1. Curva con splines y nudos",
          "2. Pesos de clase y calibración (HGB)",
          "3. Calibración Platt vs. Isotónica",
          "4. Selección de umbral con Clopper-Pearson (VAL)",
          "5. Estimación por Bootstrap en FNR (VAL)",
          "6. Bootstrap vs. Clopper-Pearson (VAL)",
      ),
  )

  # Fila 1, Col 1: Splines
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=p_grid_real,
          name="Real (simulada)",
          mode="lines",
          legendgroup="real",
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=modelo_spline.predict_proba(X_grid)[:, 1],
          name="Logística + splines",
          mode="lines",
      ),
      row=1,
      col=1,
  )
  for idx, k in enumerate(nudos_psa):
    fig.add_vline(
        x=float(k),
        line_dash="dot",
        line_color="gray",
        row=1,
        col=1,
        annotation_text=f"n{idx + 1}" if idx < 3 else "",
        annotation_position="top",
    )

  # Fila 1, Col 2: Pesos y calibración
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=p_grid_real,
          name="Real (simulada)",
          mode="lines",
          legendgroup="real",
          showlegend=False,
      ),
      row=1,
      col=2,
  )
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=hgb_base.predict_proba(X_grid)[:, 1],
          name="HGB pesos (sin calibrar)",
          mode="lines",
          line=dict(dash="dash"),
      ),
      row=1,
      col=2,
  )
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=hgb_platt.predict_proba(X_grid)[:, 1],
          name="HGB calibrado Platt",
          mode="lines",
          legendgroup="platt",
      ),
      row=1,
      col=2,
  )

  # Fila 1, Col 3: Platt vs Isotónica
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=p_grid_real,
          name="Real (simulada)",
          mode="lines",
          legendgroup="real",
          showlegend=False,
      ),
      row=1,
      col=3,
  )
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=hgb_platt.predict_proba(X_grid)[:, 1],
          name="HGB Platt (sigmoid)",
          mode="lines",
          legendgroup="platt",
          showlegend=False,
      ),
      row=1,
      col=3,
  )
  fig.add_trace(
      go.Scatter(
          x=psa_grid,
          y=hgb_iso.predict_proba(X_grid)[:, 1],
          name="HGB Isotónica",
          mode="lines",
          line=dict(dash="dashdot"),
      ),
      row=1,
      col=3,
  )

  # Fila 2, Col 1: Selección umbral con Clopper-Pearson
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["fn_pct"],
          name="FNR: FN entre positivos (val)",
          mode="lines",
          legendgroup="fnr",
      ),
      row=2,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["ci95_sup_pct"],
          name="Límite sup. exacto Clopper-Pearson 95% (val)",
          mode="lines",
          line=dict(dash="dash"),
          legendgroup="cp",
      ),
      row=2,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"], y=df_umb["fp_pct"], name="% FP (val)", mode="lines"
      ),
      row=2,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["deriv_pct"],
          name="% Derivaciones (val)",
          mode="lines",
          line=dict(dash="dot"),
      ),
      row=2,
      col=1,
  )
  fig.add_hline(
      y=1.0,
      line_dash="dashdot",
      line_color="black",
      row=2,
      col=1,
      annotation_text="Techo FN (1%)",
  )
  if umbral_elegido is not None:
    fig.add_vline(
        x=float(umbral_elegido),
        line_color="green",
        row=2,
        col=1,
        annotation_text=f"τ={umbral_elegido:.3f}",
    )

  # Fila 2, Col 2: Bootstrap en FNR
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["fn_pct"],
          name="FNR observada (val)",
          mode="lines",
          legendgroup="fnr",
          showlegend=False,
      ),
      row=2,
      col=2,
  )
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["boot_ci95_sup_pct"],
          name="Límite sup. Bootstrap 95% (val)",
          mode="lines",
          line=dict(dash="dot", color="purple"),
          legendgroup="boot",
      ),
      row=2,
      col=2,
  )
  fig.add_hline(
      y=1.0,
      line_dash="dashdot",
      line_color="black",
      row=2,
      col=2,
      annotation_text="Techo FN (1%)",
  )

  # Fila 2, Col 3: Comparativa Bootstrap vs. Clopper-Pearson
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["fn_pct"],
          name="FNR observada (val)",
          mode="lines",
          legendgroup="fnr",
          showlegend=False,
      ),
      row=2,
      col=3,
  )
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["boot_ci95_sup_pct"],
          name="Bootstrap IC 95% sup.",
          mode="lines",
          line=dict(dash="dot", color="purple"),
          legendgroup="boot",
          showlegend=False,
      ),
      row=2,
      col=3,
  )
  fig.add_trace(
      go.Scatter(
          x=df_umb["umbral"],
          y=df_umb["ci95_sup_pct"],
          name="Clopper-Pearson IC 95% sup.",
          mode="lines",
          line=dict(dash="dash", color="darkorange"),
          legendgroup="cp",
          showlegend=False,
      ),
      row=2,
      col=3,
  )
  fig.add_hline(
      y=1.0,
      line_dash="dashdot",
      line_color="black",
      row=2,
      col=3,
      annotation_text="Techo FN (1%)",
  )

  for c in range(1, 4):
    fig.update_xaxes(title_text="PSA total (ng/mL)", row=1, col=c)
    fig.update_yaxes(title_text="Probabilidad a 5 años", row=1, col=c)
    fig.update_xaxes(title_text="Umbral de probabilidad (τ)", row=2, col=c)

  fig.update_yaxes(
      title_text="Porcentaje (%; FNR sobre positivos, FP/derivaciones sobre cohorte)",
      row=2,
      col=1,
  )
  fig.update_yaxes(title_text="FNR / Límite superior (%)", row=2, col=2)
  fig.update_yaxes(title_text="FNR / Límite superior (%)", row=2, col=3)

  fig.update_layout(
      title=(
          "Prototipo interactivo — Cribado de cáncer de próstata a 5 años"
          " (simulación)"
      ),
      height=900,
      hovermode="x unified",
      template="plotly_white",
  )

  plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
  page = f"""<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Simulación sintética: calibración y selección de umbral</title>
    <style>
        body {{ margin: 0; color: #1f2937; font-family: Arial, sans-serif; line-height: 1.5; }}
        header {{ max-width: 1200px; margin: 0 auto; padding: 28px 24px 12px; }}
        h1 {{ margin: 0 0 12px; font-size: clamp(1.6rem, 3vw, 2.2rem); }}
        header p {{ margin: 8px 0; }}
        .legend {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px 24px; padding-left: 22px; }}
        .links {{ margin-top: 16px; }}
        a {{ color: #075985; }}
    </style>
</head>
<body>
    <header>
        <h1>Simulación sintética: calibración y selección de umbral</h1>
        <p>Esta simulación permite validar la lógica del pipeline y comparar modelos, calibración y umbrales con datos sintéticos.</p>
        <p>No representa la serie histórica de 2006–2021 ni es una herramienta clínica para tomar decisiones.</p>
        <h2>Qué muestra cada gráfica</h2>
        <ul class="legend">
            <li><strong>Curva con splines:</strong> relación entre PSA y probabilidad estimada por el modelo logístico.</li>
            <li><strong>Pesos de clase y calibración:</strong> comparación del modelo de boosting antes y después de ponderar y calibrar.</li>
            <li><strong>Platt frente a isotónica:</strong> comparación de los métodos de calibración de probabilidades.</li>
            <li><strong>Selección de umbral (Clopper-Pearson):</strong> FNR y límite exacto Clopper-Pearson, junto con FP y derivaciones.</li>
            <li><strong>Bootstrap en FNR:</strong> límite superior del IC 95% estimado mediante remuestreo bootstrap.</li>
            <li><strong>Bootstrap vs. Clopper-Pearson:</strong> comparación directa de ambos métodos de incertidumbre sobre la FNR.</li>
        </ul>
        <p class="links"><a href="https://github.com/100015152-svg/cribado-prostata-crisp-dm/tree/cribado-prostata-crisp-dm">Ver el repositorio del proyecto en GitHub</a></p>
    </header>
    {plot_html}
</body>
</html>
"""
  return guardar_texto_seguro(page, output_html, fallback_dir)


def main() -> None:
  n = 5000
  seed = 42
  rng = np.random.default_rng(seed)

  base_dir = Path.cwd().resolve()
  output_dir = (base_dir / "outputs").resolve()

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

  n_positivos_val = int(y_val.sum())
  minimo_ci95_val = float(beta.ppf(0.975, 1, n_positivos_val) * 100.0)
  print(
      f"\nCasos positivos en VAL: {n_positivos_val} | "
      f"Límite sup. exacto Clopper-Pearson 95% incluso con 0 FN: {minimo_ci95_val:.2f}%"
  )

  umbrales = np.linspace(0.005, 0.15, 80)
  filas_val = [
      evaluar_umbral(
          y_val.values,
          preds_val["HGB calibrado Platt"],
          tau,
          rng=rng,
          n_boot=500,
      )
      for tau in umbrales
  ]
  df_umb = pd.DataFrame(filas_val)

  validos = df_umb[df_umb["ci95_sup_pct"] < 1.0]
  umbral_elegido = float(validos["umbral"].max()) if not validos.empty else None

  res_base_test = evaluar_umbral(
      y_test.values, test["psa_total"].values, 3.0, rng=rng, n_boot=500
  )
  base_fp = res_base_test["fp_pct"]
  base_fn = res_base_test["fn_pct"]
  base_fn_ci95 = res_base_test["ci95_sup_pct"]
  base_fn_boot = res_base_test["boot_ci95_sup_pct"]
  base_deriv = res_base_test["deriv_pct"]

  print(
      f"\nProtocolo base en TEST (PSA >= 3.0) -> Derivaciones: {base_deriv:.2f}% | "
      f"FP: {base_fp:.2f}% | FNR: {base_fn:.2f}% "
      f"(Clopper-Pearson sup.: {base_fn_ci95:.2f}% | Bootstrap sup.: {base_fn_boot:.2f}%)"
  )

  if umbral_elegido is not None:
    res_test = evaluar_umbral(
        y_test.values,
        preds_test["HGB calibrado Platt"],
        umbral_elegido,
        rng=rng,
        n_boot=500,
    )
    red_fp = ((base_fp - res_test["fp_pct"]) / max(base_fp, 1e-9)) * 100.0
    red_deriv = (
        (base_deriv - res_test["deriv_pct"]) / max(base_deriv, 1e-9)
    ) * 100.0
    print(f"Umbral elegido (seleccionado en VAL): {umbral_elegido:.4f}")
    print(
        f"En TEST con umbral elegido -> Derivaciones: {res_test['deriv_pct']:.2f}% | "
        f"FP: {res_test['fp_pct']:.2f}% | FNR: {res_test['fn_pct']:.2f}% "
        f"(Clopper-Pearson sup.: {res_test['ci95_sup_pct']:.2f}% | "
        f"Bootstrap sup.: {res_test['boot_ci95_sup_pct']:.2f}%)"
    )
    print(
        f"Reducción relativa en TEST vs. protocolo base -> Derivaciones: {red_deriv:.1f}% | "
        f"FP: {red_fp:.1f}%"
    )
  else:
    print(
        "Ningún umbral en el rango evaluado cumple límite superior exacto "
        "Clopper-Pearson 95% de FNR < 1% en VAL."
    )

  png_guardado = generar_graficas_png(
      df_umb=df_umb,
      train_psa=train["psa_total"],
      modelo_spline=modelo_spline,
      hgb_base=hgb_base,
      hgb_platt=hgb_platt,
      hgb_iso=hgb_iso,
      umbral_elegido=umbral_elegido,
      output_path=output_dir / "todas_las_graficas_prostata.png",
      fallback_dir=base_dir,
  )
  html_guardado = generar_graficas_html(
      df_umb=df_umb,
      train_psa=train["psa_total"],
      modelo_spline=modelo_spline,
      hgb_base=hgb_base,
      hgb_platt=hgb_platt,
      hgb_iso=hgb_iso,
      umbral_elegido=umbral_elegido,
      output_html=output_dir / "graficas_interactivas_prostata.html",
      fallback_dir=base_dir,
  )
  print(f"\nGuardadas figuras en: {png_guardado} y {html_guardado}")


if __name__ == "__main__":
  main()