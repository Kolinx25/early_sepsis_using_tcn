"""Phase 1 — Rigorous dataset audit and exploratory analysis.

Generates every figure/table required by the SepsiGuard research protocol:
dimensions, distributions, prevalence, missingness (incl. informative-
missingness hypothesis tests), temporal measurement frequency, outliers /
clinical plausibility, and training set A vs B differences.

Outputs
    results/figures/exploratory_analysis/*.png
    results/figures/missingness/*.png
    results/tables/dataset_summary/*.csv
    results/tables/descriptive_statistics/*.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.load import CLINICAL_CHANNELS, load_observations

ROOT = Path(__file__).resolve().parents[2]
FIG_EDA = ROOT / "results" / "figures" / "exploratory_analysis"
FIG_MIS = ROOT / "results" / "figures" / "missingness"
TAB_DS = ROOT / "results" / "tables" / "dataset_summary"
TAB_DESC = ROOT / "results" / "tables" / "descriptive_statistics"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

# Physiologically plausible ranges (clinical literature / challenge docs)
PLAUSIBLE = {
    "HR": (10, 300), "O2Sat": (50, 100), "Temp": (30, 43), "SBP": (30, 300),
    "MAP": (20, 250), "DBP": (10, 200), "Resp": (4, 80), "EtCO2": (10, 120),
    "BaseExcess": (-30, 30), "HCO3": (5, 60), "FiO2": (0.1, 1.0), "pH": (6.5, 8.0),
    "PaCO2": (10, 200), "SaO2": (30, 100), "AST": (0, 20000), "BUN": (0, 400),
    "Alkalinephos": (0, 5000), "Calcium": (0, 20), "Chloride": (70, 200),
    "Creatinine": (0, 40), "Bilirubin_direct": (0, 60), "Glucose": (0, 2000),
    "Lactate": (0, 30), "Magnesium": (0.5, 10), "Phosphate": (0.1, 30),
    "Potassium": (1, 12), "Bilirubin_total": (0, 100), "TroponinI": (0, 100),
    "Hct": (8, 70), "Hgb": (2, 25), "PTT": (10, 200), "WBC": (0, 200),
    "Fibrinogen": (20, 1500), "Platelets": (1, 2000),
    "Age": (15, 110), "Gender": (0, 1), "Unit1": (0, 1), "Unit2": (0, 1),
}


def run_audit() -> dict:
    df = load_observations()
    pat = pd.read_parquet(ROOT / "data" / "interim" / "validated" / "patients.parquet")
    for d in (FIG_EDA, FIG_MIS, TAB_DS, TAB_DESC):
        d.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}

    # ------------------------------------------------------------------ counts
    summary["total_patients"] = int(df.patient_id.nunique())
    summary["total_hourly_rows"] = int(len(df))
    summary["patients_setA"] = int((pat.dataset == "training_setA").sum())
    summary["patients_setB"] = int((pat.dataset == "training_setB").sum())
    summary["septic_patients"] = int(pat.septic.sum())
    summary["patient_level_sepsis_prevalence"] = round(float(pat.septic.mean()), 4)
    summary["hourly_label_prevalence"] = round(float(df.SepsisLabel.mean()), 4)
    summary["median_stay_hours"] = float(pat.n_hours.median())
    summary["mean_stay_hours"] = round(float(pat.n_hours.mean()), 2)

    # ---- window-eligibility (historical verification) -----------------------
    # Policy (Phase 3): a prediction window at hour t requires a complete
    # 12 h history (t >= 11), at-risk status (t < onset) and a complete 12 h
    # future horizon (t <= stay_end - 12). Patients with >= 24 hourly records
    # are therefore eligible; this reproduces the historical 30,661 figure.
    eligible = pat[pat.n_hours >= 24]
    summary["window_eligible_patients_ge24h"] = int(len(eligible))
    summary["total_potential_windows"] = int((pat.n_hours - 23).clip(lower=0).sum())
    septic_eligible = eligible[eligible.septic == 1]
    summary["window_eligible_septic_patients"] = int(len(septic_eligible))

    # ---------------------------------------------------------------- onset
    onset = pat.loc[pat.septic == 1, "onset_hour"]
    summary["median_onset_hour"] = float(onset.median())
    summary["sepsis_onset_within_6h_of_admission"] = round(
        float((onset <= 6).mean()), 4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(pat.n_hours, bins=np.arange(0, 350, 6), color="#2563eb", alpha=0.8)
    axes[0].set(xlabel="ICU stay (hours)", ylabel="patients", title="ICU stay length")
    axes[0].hist(pat.loc[pat.septic == 1, "n_hours"], bins=np.arange(0, 350, 6),
                  color="#dc2626", alpha=0.8, label="septic")
    axes[0].legend()
    axes[1].hist(onset, bins=48, color="#dc2626", alpha=0.8)
    axes[1].set(xlabel="hour of sepsis onset", ylabel="patients",
                title=f"Sepsis onset time (n={len(onset)}, median={onset.median():.0f} h)")
    fig.tight_layout()
    fig.savefig(FIG_EDA / "fig1_stay_length_and_onset.png")
    plt.close(fig)

    # Hourly label distribution: consecutive-positive runs
    runs = []
    for _, g in df.groupby("patient_id", sort=False):
        lab = g.SepsisLabel.values
        idx = np.flatnonzero(np.diff(lab) == 1) + 1
        starts = np.r_[0, idx]
        ends = np.r_[idx, len(lab)]
        for s, e in zip(starts, ends):
            if lab[s] == 1:
                runs.append(e - s)
    summary["median_positive_run_hours"] = float(np.median(runs))
    summary["frac_positive_hours_in_multi_hour_runs"] = round(
        float(sum(r for r in runs if r > 1) / sum(runs)), 4)

    # ------------------------------------------------------- descriptive stats
    stats_rows = []
    for ch in CLINICAL_CHANNELS + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]:
        s = df[ch].dropna()
        stats_rows.append({
            "channel": ch, "n_observed": int(len(s)),
            "pct_missing": round(float(1 - len(s) / len(df)), 4),
            "mean": round(float(s.mean()), 3), "sd": round(float(s.std()), 3),
            "median": round(float(s.median()), 3),
            "p5": round(float(s.quantile(0.05)), 3),
            "p95": round(float(s.quantile(0.95)), 3),
            "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
        })
    desc = pd.DataFrame(stats_rows)
    desc.to_csv(TAB_DESC / "table_descriptive_statistics.csv", index=False)

    # clinical plausibility / outliers
    plaus = []
    for ch in CLINICAL_CHANNELS:
        lo, hi = PLAUSIBLE[ch]
        s = df[ch].dropna()
        out = ((s < lo) | (s > hi)).mean()
        plaus.append({"channel": ch, "plausible_range": f"[{lo}, {hi}]",
                      "pct_outside_range": round(float(out) * 100, 4)})
    pd.DataFrame(plaus).to_csv(TAB_DESC / "table_clinical_plausibility.csv", index=False)
    worst = pd.DataFrame(plaus).sort_values("pct_outside_range", ascending=False).head(3)
    summary["worst_plausibility_channels"] = {
        r.channel: r.pct_outside_range for r in worst.itertuples()}

    # ------------------------------------------------------------- missingness
    miss = desc.set_index("channel")["pct_missing"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    miss.plot.barh(ax=ax, color="#7c3aed", alpha=0.85)
    ax.set(xlabel="% of hourly records missing", title="Missingness per clinical channel")
    fig.tight_layout()
    fig.savefig(FIG_MIS / "fig2_missingness_per_channel.png")
    plt.close(fig)
    desc[["channel", "pct_missing"]].rename(
        columns={"pct_missing": "pct_missing"}).to_csv(
        TAB_DS / "table_missingness_by_channel.csv", index=False)

    # missingness vs stay position (decile of ICULOS)
    df_pos = df.copy()
    df_pos["stay_len"] = df_pos.groupby("patient_id")["ICULOS"].transform("max")
    df_pos["pos_frac"] = df_pos.ICULOS / df_pos.stay_len.clip(lower=1)
    bins = pd.cut(df_pos.pos_frac, bins=10, labels=False)
    sample_cols = ["HR", "SBP", "Lactate", "BUN", "WBC", "Platelets", "Glucose", "Temp"]
    rates = 1 - df_pos.groupby(bins)[sample_cols].apply(lambda g: g.mean())
    rates.index = [f"{int(i)*10}-{int(i)*10+10}%" for i in rates.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    rates.plot(ax=ax)
    ax.set(xlabel="position within ICU stay (deciles)", ylabel="% missing",
           title="Missingness evolves over the stay (informative missingness)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_MIS / "fig3_missingness_by_stay_position.png")
    plt.close(fig)

    # ------------------------------------------- informative missingness tests
    # (1) patient-level observation rates: septic vs non-septic (MW U)
    per_pat = df.groupby("patient_id")[CLINICAL_CHANNELS].apply(lambda g: g.notna().mean())
    septic_ids = set(pat.loc[pat.septic == 1, "patient_id"])
    per_pat["septic"] = per_pat.index.map(lambda p: int(p in septic_ids))
    rows = []
    for ch in CLINICAL_CHANNELS:
        a = per_pat.loc[per_pat.septic == 1, ch]
        b = per_pat.loc[per_pat.septic == 0, ch]
        u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial effect size
        ra, rb = stats.rankdata(a), stats.rankdata(b)
        rbc = 1 - 2 * (ra.sum() - ra.size * (ra.size + 1) / 2) / (ra.size * rb.size)
        rows.append({"channel": ch,
                     "obs_rate_septic": round(float(a.mean()), 4),
                     "obs_rate_nonseptic": round(float(b.mean()), 4),
                     "rank_biserial": round(float(rbc), 4),
                     "p_value": float(pval)})
    mw = pd.DataFrame(rows)
    mw["p_bonferroni"] = np.minimum(mw.p_value * len(mw), 1.0)
    mw.to_csv(TAB_DESC / "table_missingness_sepsis_tests.csv", index=False)
    summary["channels_missingness_differs_after_bonferroni"] = int(
        (mw.p_bonferroni < 0.05).sum())

    top = mw.reindex(mw.rank_biserial.abs().sort_values(ascending=False).index).head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(top))
    ax.bar(x - 0.2, top.obs_rate_septic, 0.4, label="septic patients", color="#dc2626", alpha=0.85)
    ax.bar(x + 0.2, top.obs_rate_nonseptic, 0.4, label="non-septic patients", color="#0891b2", alpha=0.85)
    ax.set_xticks(x, top.channel, rotation=30, ha="right")
    ax.set(ylabel="mean hourly observation rate", ylim=(0, 1),
           title="Observation rates differ by eventual sepsis status (top 8 channels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_MIS / "fig4_informative_missingness_septic_vs_nonseptic.png")
    plt.close(fig)

    # (2) prediction-signal check: current-hour missingness indicators only
    rng = np.random.default_rng(42)
    sub_ids = rng.choice(np.array(sorted(set(df.patient_id))), size=4000, replace=False)
    sub = df[df.patient_id.isin(sub_ids)]
    onset_map = pat.set_index("patient_id")["onset_hour"]
    stay_map = pat.set_index("patient_id").n_hours
    X_miss = sub[CLINICAL_CHANNELS].isna().astype(np.float32).values
    t = sub.ICULOS.values
    onset_t = sub.patient_id.map(onset_map).values.astype(float)
    stay_t = sub.patient_id.map(stay_map).values
    at_risk = np.isfinite(onset_t) & (t < onset_t) | ~np.isfinite(onset_t)
    horizon_ok = t <= (stay_t - 12)
    y12 = (np.isfinite(onset_t) & (t < onset_t) & (onset_t <= t + 12)).astype(int)
    keep = at_risk & horizon_ok
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_miss[keep], y12[keep])
    auroc = roc_auc_score(y12[keep], m.predict_proba(X_miss[keep])[:, 1])
    summary["auroc_y12_from_missingness_indicators_alone"] = round(float(auroc), 4)
    summary["n_at_risk_rows_in_missingness_check"] = int(keep.sum())
    summary["y12_positive_rate_at_risk_rows"] = round(float(y12[keep].mean()), 4)

    # ------------------------------------------------------------- vitals dist
    vitals = ["HR", "SBP", "DBP", "Resp", "Temp", "O2Sat", "MAP"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, ch in zip(axes.ravel(), vitals + ["Lactate"]):
        s = df[ch].dropna()
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        sep = s[df.loc[s.index, "patient_id"].map(
            lambda p: p in septic_ids)]
        ax.hist(np.clip(s, lo, hi), bins=bins, color="#0891b2", alpha=0.6, density=True, label="all")
        ax.hist(np.clip(sep, lo, hi), bins=bins, color="#dc2626", alpha=0.5, density=True, label="septic")
        ax.set(title=ch, yticks=[])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Observed clinical distributions (0.5–99.5 pct range)")
    fig.tight_layout()
    fig.savefig(FIG_EDA / "fig5_clinical_distributions.png")
    plt.close(fig)

    # ---------------------------------------------------------- set A vs set B
    comp = []
    for ch in CLINICAL_CHANNELS + ["Age", "HospAdmTime"]:
        a = df.loc[df.dataset == "training_setA", ch].dropna()
        b = df.loc[df.dataset == "training_setB", ch].dropna()
        _, p = stats.mannwhitneyu(a.sample(min(len(a), 100_000), random_state=0),
                                  b.sample(min(len(b), 100_000), random_state=0))
        comp.append({"channel": ch, "mean_A": round(float(a.mean()), 3),
                     "mean_B": round(float(b.mean()), 3),
                     "median_A": round(float(a.median()), 3),
                     "median_B": round(float(b.median()), 3),
                     "p_value": float(p)})
    comp = pd.DataFrame(comp)
    comp["p_bonferroni"] = np.minimum(comp.p_value * len(comp), 1.0)
    comp.to_csv(TAB_DS / "table_setA_vs_setB.csv", index=False)
    tab = pd.crosstab(pat.dataset, pat.septic)
    chi2, pchi, _, _ = stats.chi2_contingency(tab)
    summary["setA_vs_setB_sepsis_prevalence_chi2_p"] = float(pchi)
    summary["prevalence_A"] = round(float(tab.loc["training_setA", 1] / tab.loc["training_setA"].sum()), 4)
    summary["prevalence_B"] = round(float(tab.loc["training_setB", 1] / tab.loc["training_setB"].sum()), 4)
    # hospital unit encoding differs between sets
    summary["unit1_share_A"] = round(float(df.loc[df.dataset == "training_setA", "Unit1"].mean()), 4)
    summary["unit1_share_B"] = round(float(df.loc[df.dataset == "training_setB", "Unit1"].mean()), 4)
    summary["channels_significantly_different_A_vs_B_bonferroni"] = int((comp.p_bonferroni < 0.05).sum())

    # ---------------------------------------------- temporal measurement freq
    gap_rows = []
    for ch in ["HR", "SBP", "Temp", "O2Sat", "Lactate", "BUN", "WBC", "Glucose"]:
        s = df.loc[df[ch].notna(), ["patient_id", "ICULOS"]].copy() if ch == "HR" else None
        s = df.loc[df[ch].notna(), ["patient_id", "ICULOS"]]
        gaps = s.groupby("patient_id")["ICULOS"].apply(lambda x: x.diff().dropna())
        gap_rows.append({"channel": ch,
                         "median_hours_between_observations": float(gaps.median()),
                         "p90_hours_between_observations": float(gaps.quantile(0.9))})
    gdf = pd.DataFrame(gap_rows)
    gdf.to_csv(TAB_DS / "table_measurement_frequency.csv", index=False)

    # ------------------------------------------------------------ summary save
    with open(TAB_DS / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    (TAB_DS / "table_dataset_characteristics.csv").write_text(
        "characteristic,value\n" + "\n".join(
            f"{k},{v}" for k, v in summary.items()))
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run_audit()