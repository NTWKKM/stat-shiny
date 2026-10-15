# Standalone Publication Figure Reference (Cloud Sandbox)

In cloud sandboxes (e.g. Claude Web Analysis Tool), `medstat` is not installed as an external package. All publication figures are generated using standard Python libraries (`matplotlib.pyplot`, `numpy`, `pandas`) and saved directly at 300 DPI with zero custom dependencies.

---

## 1. Architectural Principles & Standards

### Publication Specifications

- **Resolution**: 300 DPI strictly enforced across all exported images (`dpi=300`).
- **Dimensions**:
  - `docx` target: Width 6.5 in (fits standard 8.5" x 11" page with 1.0" margins).
  - `pptx` target: Width 11.5–12.0 in (fits 16:9 widescreen slides).
  - `web` target: Max width 900 px with responsive PNG rendering.
- **Colorblind-Safe Palettes**: Okabe-Ito / Tol palettes (`#0072B2`, `#D55E00`, `#009E73`, `#CC79A7`, `#E69F00`, `#56B4E9`).
- **Thai Typography Support**: Fallback font family checks (`Sarabun`, `Thonburi`, `Sukhumvit Set`, `sans-serif`) prevent square-box glyph errors.
- **Zero-PHI Guarantee**: Axis labels, legends, and annotations strictly prohibit raw patient identifiers (HN, citizen IDs, patient names).

---

## 2. Standalone Matplotlib Recipes (Accepting Recipe Outputs)

The functions below are 100% self-contained and directly accept the output structures returned by `python-recipes.md` Recipes 3–7 and 11.

### 1. Forest Plot (`plot_forest`)

Accepts output from Recipe 7 (`fit_firth_logistic`) or Recipe 11 (`fit_survival_analysis_suite`): `model_df` DataFrame containing `['term', 'estimate', 'ci_lower', 'ci_upper', 'p_value']`.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forest(
    model_df: pd.DataFrame,
    scale: str = "OR",
    title: str = "Multivariable Predictors (Forest Plot)",
    out_path: str = "figures/forest.png",
) -> str:
    """Standalone 300 DPI publication forest plot."""
    is_ratio = scale.upper() in ("OR", "HR", "RR")
    null_val = 1.0 if is_ratio else 0.0

    df = model_df.copy().reset_index(drop=True)
    y_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.5, max(4.0, len(df) * 0.45)))
    ax.axvline(
        x=null_val, color="#94A3B8", linestyle="--", linewidth=1.2, zorder=1
    )

    for idx, row in df.iterrows():
        est = float(row["estimate"])
        cil = float(row["ci_lower"])
        ciu = float(row["ci_upper"])
        ax.plot([cil, ciu], [idx, idx], color="#0284C7", linewidth=2.0, zorder=2)
        ax.scatter(
            [est],
            [idx],
            color="#0284C7",
            s=45,
            edgecolors="#0F172A",
            zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["term"].astype(str), fontsize=10)
    ax.invert_yaxis()
    if is_ratio:
        ax.set_xscale("log")
    ax.set_xlabel(f"Effect Size ({scale}, 95% CI)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 2. Kaplan-Meier Survival Curve (`plot_kaplan_meier`)

Accepts `km_curves` dictionary from Recipe 11 (`fit_survival_analysis_suite`).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_kaplan_meier(
    km_data: dict,
    title: str = "Kaplan-Meier Survival Curves",
    x_label: str = "Time (Days)",
    out_path: str = "figures/km_survival.png",
) -> str:
    """Standalone 300 DPI publication Kaplan-Meier survival curves with 95% CI bands."""
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for idx, (group_name, curve) in enumerate(km_data.items()):
        color = palette[idx % len(palette)]
        timeline = np.asarray(curve.get("timeline", []), dtype=float)
        survival = np.asarray(curve.get("survival", []), dtype=float)

        ax.step(
            timeline,
            survival,
            where="post",
            label=str(group_name),
            color=color,
            linewidth=2.0,
        )

        if "ci_lower" in curve and "ci_upper" in curve:
            cil = np.asarray(curve["ci_lower"], dtype=float)
            ciu = np.asarray(curve["ci_upper"], dtype=float)
            if len(cil) == len(timeline) and len(ciu) == len(timeline):
                ax.fill_between(
                    timeline, cil, ciu, step="post", alpha=0.18, color=color
                )

        raw_cens = curve.get("censored_times")
        if raw_cens is not None and len(raw_cens) > 0 and len(timeline) > 0:
            for ct in raw_cens:
                ct_val = float(ct)
                idx_t = np.searchsorted(timeline, ct_val, side="right") - 1
                if 0 <= idx_t < len(survival):
                    ax.plot(
                        ct_val,
                        survival[idx_t],
                        marker="+",
                        color=color,
                        markersize=6,
                        mew=1.5,
                    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Survival Probability", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, loc="lower left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 3. Empirical ROC Curve & DeLong CI (`plot_roc_curve`)

Accepts `roc_data` dictionary from Recipe 3 (`calculate_roc_curve` or `auc_ci_delong`).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(
    roc_data: dict,
    title: str = "Receiver Operating Characteristic (ROC)",
    out_path: str = "figures/roc_curve.png",
) -> str:
    """Standalone 300 DPI publication ROC curve."""
    fpr = np.asarray(roc_data.get("fpr", []), dtype=float)
    tpr = np.asarray(roc_data.get("tpr", []), dtype=float)
    auc = float(roc_data.get("auc", 0.5))
    ci_l = roc_data.get("ci_lower")
    ci_u = roc_data.get("ci_upper")

    label = (
        f"AUC = {auc:.3f} (95% CI {float(ci_l):.3f}–{float(ci_u):.3f})"
        if (ci_l is not None and ci_u is not None)
        else f"AUC = {auc:.3f}"
    )

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Chance")
    ax.plot(fpr, tpr, color="#0072B2", linewidth=2.2, label=label)

    opt_pt = roc_data.get("optimal_point")
    if opt_pt:
        opt_fpr = float(opt_pt.get("fpr", 0.0))
        opt_tpr = float(opt_pt.get("tpr", 0.0))
        cut = float(opt_pt.get("cutoff", 0.0))
        ax.scatter(
            [opt_fpr],
            [opt_tpr],
            color="#D55E00",
            s=60,
            zorder=4,
            label=f"Cutoff = {cut:.2f}",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sensitivity (TPR)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 4. Calibration Curve (`plot_calibration`)

Accepts decile calibration data from Recipe 4 (`evaluate_calibration`).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_calibration(
    calib_bins: dict,
    title: str = "Model Calibration Curve",
    brier_score: float | None = None,
    ici: float | None = None,
    out_path: str = "figures/calibration.png",
) -> str:
    """Standalone 300 DPI publication decile calibration curve."""
    pred_mean = np.asarray(calib_bins.get("pred_mean", []), dtype=float)
    obs_rate = np.asarray(calib_bins.get("obs_rate", []), dtype=float)
    ci_l = np.asarray(calib_bins.get("ci_lower", obs_rate), dtype=float)
    ci_u = np.asarray(calib_bins.get("ci_upper", obs_rate), dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94A3B8",
        label="Perfect Calibration",
    )

    yerr = [obs_rate - ci_l, ci_u - obs_rate]
    ax.errorbar(
        pred_mean,
        obs_rate,
        yerr=yerr,
        fmt="o-",
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=1.5,
        capsize=3,
        linewidth=1.8,
        label="Deciles (95% CI)",
    )

    annot_lines = []
    if brier_score is not None:
        annot_lines.append(f"Brier: {brier_score:.3f}")
    if ici is not None:
        annot_lines.append(f"ICI: {ici:.3f}")
    if annot_lines:
        ax.text(
            0.05,
            0.92,
            "\n".join(annot_lines),
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Event Rate", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 5. Decision Curve Analysis (`plot_dca`)

Accepts DataFrame from Recipe 4 (`calculate_dca` with `['threshold', 'net_benefit', 'strategy']`).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_dca(
    dca_df: pd.DataFrame,
    title: str = "Decision Curve Analysis (Net Benefit)",
    out_path: str = "figures/dca.png",
) -> str:
    """Standalone 300 DPI publication Vickers Decision Curve Analysis."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    palette = {
        "Treat All": ("#64748B", ":", 1.5),
        "Treat None": ("#94A3B8", "-", 1.0),
        "Model": ("#0072B2", "-", 2.2),
    }

    for strat in dca_df["strategy"].unique():
        sub = dca_df[dca_df["strategy"] == strat].sort_values("threshold")
        color, ls, lw = palette.get(strat, ("#009E73", "-", 1.8))
        ax.plot(
            sub["threshold"],
            sub["net_benefit"],
            label=strat,
            color=color,
            linestyle=ls,
            linewidth=lw,
        )

    ax.set_xlabel("Threshold Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Benefit", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 6. Bland-Altman Agreement Plot (`plot_bland_altman`)

Accepts output dictionary from Recipe 5 (`calculate_bland_altman`).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_bland_altman(
    ba_data: dict,
    units: str = "units",
    title: str = "Bland-Altman Agreement Plot",
    out_path: str = "figures/bland_altman.png",
) -> str:
    """Standalone 300 DPI publication Bland-Altman agreement plot with Limits of Agreement."""
    means = np.asarray(ba_data.get("means", []), dtype=float)
    diffs = np.asarray(ba_data.get("diffs", []), dtype=float)
    mean_bias = float(ba_data.get("mean_diff", 0.0))
    loa_upper = float(ba_data.get("loa_upper", 0.0))
    loa_lower = float(ba_data.get("loa_lower", 0.0))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(means, diffs, color="#0072B2", alpha=0.6, s=35, edgecolors="none")
    ax.axhline(mean_bias, color="#0F172A", linestyle="-", linewidth=1.5, label=f"Mean Bias ({mean_bias:.2f})")
    ax.axhline(loa_upper, color="#D55E00", linestyle="--", linewidth=1.3, label=f"+1.96 SD ({loa_upper:.2f})")
    ax.axhline(loa_lower, color="#D55E00", linestyle="--", linewidth=1.3, label=f"-1.96 SD ({loa_lower:.2f})")

    ax.set_xlabel(f"Average of Measurements ({units})", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Difference ({units})", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 7. Love Plot (`plot_love`)

Accepts `love_plot_data` dict or `balance_df` from Recipe 6 (`match_propensity_scores`).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_love(
    love_data: dict | pd.DataFrame,
    threshold: float = 0.10,
    title: str = "Covariate Balance (Love Plot)",
    out_path: str = "figures/love_plot.png",
) -> str:
    """Standalone 300 DPI publication Austin 2009 Love plot of Standardized Mean Differences."""
    if hasattr(love_data, "columns"):
        vars_list = list(love_data["Variable"])
        pre_smd = [abs(float(x)) for x in love_data["Pre-Match SMD"]]
        post_smd = [abs(float(x)) for x in love_data["Post-Match SMD"]]
    else:
        vars_list = list(love_data.get("covariates", []))
        pre_smd = [abs(float(x)) for x in love_data.get("smd_raw", [])]
        post_smd = [abs(float(x)) for x in love_data.get("smd_matched", [])]

    y_pos = np.arange(len(vars_list))
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, len(vars_list) * 0.45)))

    ax.axvline(threshold, color="#DC2626", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.2f})")
    for i in range(len(vars_list)):
        ax.plot([pre_smd[i], post_smd[i]], [i, i], color="#CBD5E1", linewidth=1.5, zorder=1)
    ax.scatter(pre_smd, y_pos, color="#D55E00", s=50, marker="o", label="Pre-Match SMD", zorder=2)
    ax.scatter(post_smd, y_pos, color="#0072B2", s=50, marker="s", label="Post-Match SMD", zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_list, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Standardized Mean Difference (SMD)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

### 8. Participant Retention Flowchart (`plot_retention_flow`)

Accepts retention counts dictionary from Recipe 6 or cohort extraction.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_retention_flow(
    retention_data: dict,
    title: str = "Participant Retention Flow Diagram (STROBE / CONSORT)",
    out_path: str = "figures/retention_flow.png",
) -> str:
    """Standalone 300 DPI publication participant retention flowchart."""
    n_initial = retention_data.get("n_initial", 0)
    n_excluded = retention_data.get("n_excluded", 0)
    n_analyzed = retention_data.get("n_analyzed", 0)
    reasons = retention_data.get("exclusion_reasons", [])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axis("off")

    # Initial box
    box1 = patches.FancyBboxPatch((0.3, 0.72), 0.4, 0.18, boxstyle="round,pad=0.03", ec="#0F172A", fc="#F8FAFC", lw=1.5)
    ax.add_patch(box1)
    ax.text(0.5, 0.81, f"Initial Cohort Screened\nN = {n_initial:,}", ha="center", va="center", fontsize=11, fontweight="bold")

    # Excluded box
    reason_txt = "\n".join(f"• {r}" for r in reasons) if reasons else "Ineligible / missing key data"
    box_ex = patches.FancyBboxPatch((0.55, 0.42), 0.4, 0.20, boxstyle="round,pad=0.03", ec="#DC2626", fc="#FEF2F2", lw=1.5)
    ax.add_patch(box_ex)
    ax.text(0.75, 0.52, f"Excluded (n = {n_excluded:,})\n{reason_txt}", ha="center", va="center", fontsize=9, color="#991B1B")

    # Arrows
    ax.annotate("", xy=(0.5, 0.35), xytext=(0.5, 0.72), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0F172A"))
    ax.annotate("", xy=(0.55, 0.52), xytext=(0.5, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"))

    # Analyzed box
    box2 = patches.FancyBboxPatch((0.3, 0.15), 0.4, 0.18, boxstyle="round,pad=0.03", ec="#059669", fc="#F0FDF4", lw=1.5)
    ax.add_patch(box2)
    ax.text(0.5, 0.24, f"Final Analyzed Cohort\nN = {n_analyzed:,}", ha="center", va="center", fontsize=11, fontweight="bold", color="#065F46")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=16)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close()
    return str(p)
```

---

## 3. Standalone Interactive HTML Figures (Zero-Dependency Plotly Dicts)

In cloud sandboxes (such as Claude Web Analysis Tool), `medstat` is not available as an external package. Instead of importing external libraries, use the standalone pure-Python Plotly spec generators provided in [report-builder.md](report-builder.md#b1-standalone-pure-python-plotly-spec-generators-zero-dependencies). They build native JSON-serializable dictionaries (`{"data": [...], "layout": {...}}`) with zero external dependencies and pass directly into `generate_html_report(..., plotly_figs=[spec])`:

- **`build_plotly_roc_spec(roc_data, ...)`**: Empirical ROC curve with interactive threshold/sensitivity/specificity hover tooltips, chance diagonal, and optimal cutoff markers.
- **`build_plotly_km_spec(km_data, ...)`**: Step-function survival curves with 95% CI shaded ribbons and hover tooltips.
- **`build_plotly_calibration_spec(calib_bins, ...)`**: Decile calibration points with 95% Wilson CI error bars and Brier/ICI annotations.
- **`build_plotly_dca_spec(dca_data, ...)`**: Vickers Decision Curve Analysis comparing model net benefit against Treat All and Treat None.
- **`build_plotly_bland_altman_spec(ba_data, ...)`**: Paired difference plot with mean bias, 95% Limits of Agreement, and zero reference line.
- **`build_plotly_forest_spec(forest_data, ...)`**: Forest plot with log-scale axis, reference null line, and interactive point estimates with 95% CIs.
- **`build_plotly_love_spec(love_data, ...)`**: Austin 2009 Love plot comparing Pre- vs Post-match covariate balance relative to 0.10 SMD threshold.

```python
# Standalone Dual-View workflow in Cloud Sandbox:
# 1. Generate 300 DPI publication PNG via matplotlib
plot_roc_curve(roc_data, title="Biomarker Diagnostic Performance", out_path="figures/roc_curve.png")

# 2. Build standalone Plotly spec dict (no plotly library required)
roc_spec = build_plotly_roc_spec(roc_data, title="Biomarker Diagnostic Performance")

# 3. Pass both to generate_html_report for seamless Dual / Toggle View
html_report = generate_html_report(
    title="Clinical Diagnostic Study",
    results=results,
    tables={"Table 1": table1_df},
    figure_paths=["figures/roc_curve.png"],
    plotly_figs=[roc_spec],
    out_path="report.html",
)
```
