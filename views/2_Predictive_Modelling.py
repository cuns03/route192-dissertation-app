from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except Exception:
    accuracy_score = balanced_accuracy_score = brier_score_loss = None
    confusion_matrix = f1_score = log_loss = precision_score = None
    recall_score = roc_auc_score = roc_curve = None


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Predictive Modelling | Route 192",
    page_icon="🚌",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "model_outputs"

CV_PATH = MODEL_DIR / "baseline_cv_results.csv"
TEST_PATH = MODEL_DIR / "baseline_test_results.csv"
COMPARE_PATH = MODEL_DIR / "health_model_comparison.csv"
SHAP_BASE_PATH = MODEL_DIR / "shap_baseline.csv"
SHAP_HEALTH_PATH = MODEL_DIR / "shap_health_model_c.csv"
PRED_PATH = MODEL_DIR / "test_predictions.csv"

BEE_YELLOW = "#FFD800"
BLACK = "#161616"
OFF_WHITE = "#F7F7FC"
CARD = "#FFFFFF"
BORDER = "#DCE2F3"
GREY = "#6B7280"
BLUE = "#0053DB"
RED = "#BA1A1A"


# =============================================================================
# FINAL DISSERTATION FALLBACK VALUES
# These are used only if a local export cannot be parsed.
# =============================================================================
FINAL_CV = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
        "Log Loss": [0.6369, 0.5409, 0.5977, 0.5357],
        "Log Loss SD": [0.0032, 0.0065, 0.0030, 0.0062],
        "Brier Score": [0.2224, 0.1841, 0.2053, 0.1817],
        "Brier Score SD": [0.0009, 0.0027, 0.0014, 0.0026],
        "ROC-AUC": [0.6773, 0.7903, 0.7468, 0.7897],
        "ROC-AUC SD": [0.0034, 0.0059, 0.0049, 0.0062],
        "Balanced Accuracy": [0.6177, 0.7059, 0.6606, 0.6955],
        "Balanced Accuracy SD": [0.0041, 0.0063, 0.0070, 0.0052],
        "Precision": [0.6535, 0.7568, 0.6810, 0.7157],
        "Precision SD": [0.0040, 0.0037, 0.0061, 0.0037],
        "Recall": [0.7770, 0.7128, 0.8361, 0.8181],
        "Recall SD": [0.0095, 0.0128, 0.0126, 0.0148],
        "Specificity": [0.4584, 0.6991, 0.4852, 0.5729],
        "Specificity SD": [0.0144, 0.0045, 0.0182, 0.0114],
    }
)

FINAL_TEST = {
    "Accuracy": 0.7174,
    "Precision": 0.7156,
    "Recall": 0.8335,
    "Specificity": 0.5650,
    "F1": 0.7701,
    "ROC-AUC": 0.7917,
    "Brier Score": 0.1804,
    "Log Loss": 0.5317,
    "TN": 3477,
    "FP": 2677,
    "FN": 1345,
    "TP": 6735,
}

FINAL_COMPARE = {
    "Cross-Validation": pd.DataFrame(
        {
            "Metric": [
                "Log Loss",
                "Brier Score",
                "ROC-AUC",
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
            ],
            "Baseline": [0.5357, 0.1817, 0.7897, 0.7121, 0.7157, 0.8181, 0.5729],
            "Health-Integrated": [0.53566, 0.18170, 0.78969, 0.7119, 0.7159, 0.8167, 0.5741],
        }
    ),
    "Untouched Test": pd.DataFrame(
        {
            "Metric": [
                "Log Loss",
                "Brier Score",
                "ROC-AUC",
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
            ],
            "Baseline": [0.5317, 0.1804, 0.7917, 0.7174, 0.7156, 0.8335, 0.5650],
            "Health-Integrated": [0.53107, 0.18013, 0.79216, 0.7185, 0.7156, 0.8366, 0.5634],
        }
    ),
}


# =============================================================================
# CSS / VISUAL SYSTEM
# =============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Public+Sans:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: "Inter", sans-serif;
    }}

    .stApp {{
        background: {OFF_WHITE};
        color: #151C27;
    }}

    h1, h2, h3, h4 {{
        font-family: "Public Sans", sans-serif !important;
        letter-spacing: -0.015em;
    }}

    .block-container {{
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    .rq-badge {{
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .35rem .7rem;
        border-radius: .35rem;
        background: {BEE_YELLOW};
        color: {BLACK};
        font-family: "Public Sans", sans-serif;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .05em;
        text-transform: uppercase;
    }}

    .research-note {{
        background: #F0F3FF;
        border-left: 4px solid {BLUE};
        padding: .85rem 1rem;
        border-radius: .65rem;
        font-size: .9rem;
        color: #303642;
    }}

    .method-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: .8rem;
        padding: 1rem;
        min-height: 120px;
        box-shadow: 0 1px 4px rgba(0,0,0,.03);
    }}

    .method-card-yellow {{
        background: #FFF9D8;
        border: 1px solid #F3DA42;
        border-radius: .8rem;
        padding: 1rem;
        min-height: 120px;
    }}

    .spec-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: .9rem;
        padding: 1.15rem;
        min-height: 330px;
        box-shadow: 0 1px 5px rgba(0,0,0,.035);
    }}

    .spec-card-health {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-top: 4px solid {BEE_YELLOW};
        border-radius: .9rem;
        padding: 1.15rem;
        min-height: 330px;
        box-shadow: 0 1px 5px rgba(0,0,0,.035);
    }}

    .chip {{
        display: inline-block;
        background: #F0F3FF;
        border: 1px solid #E2E8F8;
        border-radius: .35rem;
        padding: .25rem .45rem;
        margin: .14rem .12rem;
        font-size: .72rem;
        color: #303642;
    }}

    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-top: 4px solid {BEE_YELLOW};
        border-radius: .8rem;
        padding: .95rem 1rem;
        min-height: 115px;
        box-shadow: 0 1px 4px rgba(0,0,0,.03);
    }}

    .metric-label {{
        color: {GREY};
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .metric-value {{
        font-family: "Public Sans", sans-serif;
        color: #151C27;
        font-size: 2rem;
        line-height: 1.12;
        font-weight: 800;
        margin-top: .2rem;
    }}

    .metric-sub {{
        color: {GREY};
        font-size: .76rem;
        margin-top: .35rem;
    }}

    .verdict {{
        background: {BEE_YELLOW};
        color: {BLACK};
        border-radius: .9rem;
        padding: 1.05rem 1.15rem;
    }}

    .verdict h3 {{
        margin: .1rem 0 .25rem 0;
    }}

    .insight {{
        background: #F0F3FF;
        border: 1px solid #E1E8F8;
        border-radius: .75rem;
        padding: .9rem 1rem;
    }}

    .warning-note {{
        background: #FFF8E6;
        border-left: 4px solid {BEE_YELLOW};
        border-radius: .65rem;
        padding: .85rem 1rem;
    }}

    div[data-baseweb="tab-list"] {{
        gap: .25rem;
        border-bottom: 1px solid #DCE2F3;
    }}

    button[data-baseweb="tab"] {{
        font-family: "Public Sans", sans-serif;
        font-weight: 600;
        background: transparent;
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {BLACK};
        border-top: 3px solid {BEE_YELLOW};
        background: white;
    }}

    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: .7rem;
        overflow: hidden;
    }}

    div[data-testid="stSlider"] {{
        padding-top: .2rem;
    }}

    .small-muted {{
        color: {GREY};
        font-size: .78rem;
    }}

    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================
def safe_read_csv(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_column(df, candidates):
    if df is None:
        return None
    norm_map = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def numeric(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def model_name_clean(x):
    s = str(x).strip().lower()
    if "logistic" in s:
        return "Logistic Regression"
    if "random" in s and "forest" in s:
        return "Random Forest"
    if "gradient" in s or s in {"gb", "gbm"}:
        return "Gradient Boosting"
    if "xg" in s:
        return "XGBoost"
    return str(x)


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chips(items):
    return "".join(f'<span class="chip">{x}</span>' for x in items)


def standard_plot_layout(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", color="#374151", size=12),
        legend=dict(font=dict(family="Inter")),
        hoverlabel=dict(font_family="Inter"),
    )
    fig.update_xaxes(gridcolor="#E8ECF5", zeroline=False)
    fig.update_yaxes(gridcolor="#E8ECF5", zeroline=False)
    return fig


# -----------------------------------------------------------------------------
# Try to parse CV export; otherwise final dissertation values are used.
# -----------------------------------------------------------------------------
def load_cv_results():
    raw = safe_read_csv(CV_PATH)
    if raw is None or raw.empty:
        return FINAL_CV.copy(), "dissertation fallback"

    model_col = find_column(raw, ["model", "model_name", "classifier"])
    if model_col is None:
        return FINAL_CV.copy(), "dissertation fallback"

    aliases = {
        "Log Loss": ["log_loss", "logloss", "cv_log_loss", "mean_log_loss"],
        "Brier Score": ["brier", "brier_score", "cv_brier", "mean_brier"],
        "ROC-AUC": ["roc_auc", "rocauc", "auc", "cv_roc_auc"],
        "Balanced Accuracy": ["balanced_accuracy", "balanced_acc", "bal_acc"],
        "Precision": ["precision"],
        "Recall": ["recall", "sensitivity"],
        "Specificity": ["specificity"],
    }
    sd_aliases = {
        "Log Loss SD": ["log_loss_sd", "log_loss_std", "std_log_loss"],
        "Brier Score SD": ["brier_sd", "brier_std", "std_brier"],
        "ROC-AUC SD": ["roc_auc_sd", "roc_auc_std", "std_roc_auc"],
        "Balanced Accuracy SD": ["balanced_accuracy_sd", "balanced_accuracy_std", "bal_acc_sd"],
        "Precision SD": ["precision_sd", "precision_std"],
        "Recall SD": ["recall_sd", "recall_std"],
        "Specificity SD": ["specificity_sd", "specificity_std"],
    }

    out = pd.DataFrame({"Model": raw[model_col].map(model_name_clean)})

    for final_name, cands in aliases.items():
        c = find_column(raw, cands)
        if c is None:
            return FINAL_CV.copy(), "dissertation fallback"
        out[final_name] = pd.to_numeric(raw[c], errors="coerce")

    for final_name, cands in sd_aliases.items():
        c = find_column(raw, cands)
        if c is not None:
            out[final_name] = pd.to_numeric(raw[c], errors="coerce")
        else:
            fallback_map = FINAL_CV.set_index("Model")[final_name]
            out[final_name] = out["Model"].map(fallback_map)

    wanted = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
    out = out[out["Model"].isin(wanted)].copy()
    if len(out) < 4:
        return FINAL_CV.copy(), "dissertation fallback"
    out["Model"] = pd.Categorical(out["Model"], categories=wanted, ordered=True)
    out = out.sort_values("Model").reset_index(drop=True)
    out["Model"] = out["Model"].astype(str)
    return out, CV_PATH.name


def load_test_result():
    raw = safe_read_csv(TEST_PATH)
    if raw is None or raw.empty:
        return FINAL_TEST.copy(), "dissertation fallback"

    row = raw.iloc[0]
    aliases = {
        "Accuracy": ["accuracy"],
        "Precision": ["precision"],
        "Recall": ["recall", "sensitivity"],
        "Specificity": ["specificity"],
        "F1": ["f1", "f1_score"],
        "ROC-AUC": ["roc_auc", "rocauc", "auc"],
        "Brier Score": ["brier", "brier_score"],
        "Log Loss": ["log_loss", "logloss"],
    }
    out = {}
    for k, cands in aliases.items():
        c = find_column(raw, cands)
        if c is None:
            return FINAL_TEST.copy(), "dissertation fallback"
        out[k] = numeric(row[c])

    # confusion matrix is sourced from row-level predictions when available;
    # final dissertation values remain the fallback.
    out.update({k: FINAL_TEST[k] for k in ["TN", "FP", "FN", "TP"]})
    return out, TEST_PATH.name


def load_prediction_rows():
    df = safe_read_csv(PRED_PATH)
    if df is None or df.empty:
        return None, None, None

    y_col = find_column(
        df,
        [
            "y_true",
            "actual",
            "target",
            "y",
            "bus_demand_binary",
            "boarding_occurrence",
            "actual_label",
        ],
    )
    p_col = find_column(
        df,
        [
            "y_prob",
            "probability",
            "pred_proba",
            "pred_probability",
            "prediction_probability",
            "baseline_probability",
            "baseline_prob",
            "prob_1",
        ],
    )

    if y_col is None or p_col is None:
        return df, None, None

    d = df[[y_col, p_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
    d = d.dropna()
    d = d[d[y_col].isin([0, 1])]
    d = d[(d[p_col] >= 0) & (d[p_col] <= 1)]
    if d.empty:
        return df, None, None

    return d, y_col, p_col


def load_comparison(view):
    # Final dissertation values are canonical. If an exported comparison file
    # is recognisable, use it; otherwise retain the final table values.
    raw = safe_read_csv(COMPARE_PATH)
    fallback = FINAL_COMPARE[view].copy()

    if raw is None or raw.empty:
        fallback["Delta"] = fallback["Health-Integrated"] - fallback["Baseline"]
        return fallback, "dissertation fallback"

    eval_col = find_column(raw, ["evaluation", "split", "dataset", "stage"])
    model_col = find_column(raw, ["model", "specification", "variant"])

    metric_aliases = {
        "Log Loss": ["log_loss", "logloss"],
        "Brier Score": ["brier", "brier_score"],
        "ROC-AUC": ["roc_auc", "rocauc", "auc"],
        "Accuracy": ["accuracy"],
        "Precision": ["precision"],
        "Recall": ["recall", "sensitivity"],
        "Specificity": ["specificity"],
    }

    if model_col is not None:
        working = raw.copy()
        if eval_col is not None:
            if view == "Untouched Test":
                mask = working[eval_col].astype(str).str.lower().str.contains("test")
            else:
                mask = working[eval_col].astype(str).str.lower().str.contains("cv|cross")
            if mask.any():
                working = working[mask]

        rows = []
        for label in ["Baseline", "Health-Integrated"]:
            if label == "Baseline":
                m = working[model_col].astype(str).str.lower().str.contains("baseline")
            else:
                m = working[model_col].astype(str).str.lower().str.contains("health")
            if not m.any():
                return fallback.assign(
                    Delta=fallback["Health-Integrated"] - fallback["Baseline"]
                ), "dissertation fallback"
            r = working[m].iloc[0]
            rec = {"Model": label}
            for metric, cands in metric_aliases.items():
                c = find_column(working, cands)
                if c is None:
                    return fallback.assign(
                        Delta=fallback["Health-Integrated"] - fallback["Baseline"]
                    ), "dissertation fallback"
                rec[metric] = numeric(r[c])
            rows.append(rec)

        wide = pd.DataFrame(rows).set_index("Model")
        out = pd.DataFrame(
            {
                "Metric": list(metric_aliases),
                "Baseline": [wide.loc["Baseline", m] for m in metric_aliases],
                "Health-Integrated": [wide.loc["Health-Integrated", m] for m in metric_aliases],
            }
        )
        out["Delta"] = out["Health-Integrated"] - out["Baseline"]
        return out, COMPARE_PATH.name

    fallback["Delta"] = fallback["Health-Integrated"] - fallback["Baseline"]
    return fallback, "dissertation fallback"


def load_shap(path: Path):
    raw = safe_read_csv(path)
    if raw is None or raw.empty:
        return None

    feature_col = find_column(raw, ["feature", "feature_name", "variable", "predictor"])
    value_col = find_column(
        raw,
        [
            "mean_abs_shap",
            "mean_absolute_shap",
            "mean_abs_shap_value",
            "shap_importance",
            "importance",
            "value",
        ],
    )

    # Long summary export
    if feature_col is not None and value_col is not None:
        out = raw[[feature_col, value_col]].copy()
        out.columns = ["Feature", "MeanAbsSHAP"]
        out["MeanAbsSHAP"] = pd.to_numeric(out["MeanAbsSHAP"], errors="coerce")
        out = out.dropna().sort_values("MeanAbsSHAP", ascending=False)
        return out

    # Observation-level SHAP export: each numeric feature column contains SHAP values.
    numeric_df = raw.select_dtypes(include=[np.number]).copy()
    ignored = {
        "index", "row", "id", "target", "y", "ytrue", "prediction", "probability",
        "basevalue", "expectedvalue"
    }
    usable = [
        c for c in numeric_df.columns
        if norm_name(c) not in ignored and not norm_name(c).startswith("unnamed")
    ]
    if usable:
        vals = numeric_df[usable].abs().mean().sort_values(ascending=False)
        return pd.DataFrame({"Feature": vals.index, "MeanAbsSHAP": vals.values})

    return None


def feature_domain(feature):
    s = norm_name(feature)
    if any(k in s for k in ["direction", "period", "headway", "stopposition", "stopsequence", "time"]):
        return "Operational & temporal"
    if any(k in s for k in ["retail", "employment400", "education400", "healthcare400", "poi"]):
        return "Built environment"
    if any(k in s for k in ["health", "disabil", "limiting", "poorhealth"]):
        return "Health & disability"
    if any(k in s for k in ["population", "student", "age", "female", "gender"]):
        return "Demographic"
    if any(k in s for k in ["nocar", "workfromhome", "wfh"]):
        return "Mobility"
    if any(k in s for k in ["depriv", "crime", "housing", "livingenvironment", "income", "skills"]):
        return "Socioeconomic"
    return "Other"


DOMAIN_COLOURS = {
    "Operational & temporal": "#161616",
    "Built environment": "#0053DB",
    "Demographic": "#7B8794",
    "Socioeconomic": "#A47C00",
    "Mobility": "#56616F",
    "Health & disability": "#FFD800",
    "Other": "#AAB2BD",
}


def compute_threshold_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
    bal = np.nanmean([recall, specificity])

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Balanced Accuracy": bal,
    }


# =============================================================================
# DATA
# =============================================================================
cv_df, cv_source = load_cv_results()
test_result, test_source = load_test_result()
pred_df, pred_y_col, pred_p_col = load_prediction_rows()

if pred_df is not None and pred_y_col and pred_p_col:
    base_cm = compute_threshold_metrics(
        pred_df[pred_y_col].astype(int).to_numpy(),
        pred_df[pred_p_col].to_numpy(),
        0.50,
    )
    for k in ["TN", "FP", "FN", "TP"]:
        test_result[k] = base_cm[k]


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    '<span class="rq-badge">RQ1 · Incremental Predictive Value</span>',
    unsafe_allow_html=True,
)
st.title("Predictive Modelling")
st.markdown(
    "Testing whether neighbourhood health and disability information adds "
    "out-of-sample predictive value beyond conventional demand predictors."
)

st.markdown(
    """
    <div class="research-note">
    <strong>Academic scoping note.</strong>
    Predictions concern <strong>reconstructed boarding occurrence</strong> on unseen Route 192
    trips at known stops and LSOAs. The analysis is predictive rather than causal.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Method chain — use actual grouped-split row counts from the final modelling outputs.
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(
        '<div class="method-card"><div class="metric-label">Total sample</div>'
        '<div class="metric-value" style="font-size:1.45rem;">71,200</div>'
        '<div class="metric-sub">Scheduled stop visits</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="method-card"><div class="metric-label">Development</div>'
        '<div class="metric-value" style="font-size:1.45rem;">56,966</div>'
        '<div class="metric-sub">Grouped by trip_id</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        '<div class="method-card"><div class="metric-label">Cross-validation</div>'
        '<div class="metric-value" style="font-size:1.45rem;">5 folds</div>'
        '<div class="metric-sub">StratifiedGroupKFold</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        '<div class="method-card"><div class="metric-label">Selection criterion</div>'
        '<div class="metric-value" style="font-size:1.45rem;">Log loss</div>'
        '<div class="metric-sub">Pre-specified probabilistic criterion</div></div>',
        unsafe_allow_html=True,
    )
with m5:
    st.markdown(
        '<div class="method-card-yellow"><div class="metric-label">Untouched test</div>'
        '<div class="metric-value" style="font-size:1.45rem;">14,234</div>'
        '<div class="metric-sub">Held-out trips</div></div>',
        unsafe_allow_html=True,
    )

st.caption(
    "Splits and cross-validation are grouped by trip_id so sequential stop visits from the "
    "same trip do not appear across training and validation data."
)

with st.expander("Interpretation boundary"):
    st.markdown(
        """
        Headway, direction and operating period also informed the passenger-demand
        reconstruction process. RQ1 should therefore be interpreted as a test of the
        **incremental value of health information conditional on the reconstruction
        framework**, rather than as evidence of independent causal determinants of passenger
        demand.
        """
    )


# =============================================================================
# CONTROLLED MODEL COMPARISON
# =============================================================================
st.subheader("Controlled Model Comparison")
st.caption(
    "The Health-Integrated specification differs from the conventional baseline only by the "
    "addition of the LSOA-level health block."
)

left, right = st.columns(2, gap="medium")

with left:
    st.markdown(
        f"""
        <div class="spec-card">
            <h3 style="margin-top:0">Baseline Model</h3>
            <p style="color:{GREY};font-size:.86rem;">
                Conventional operational, built-environment, demographic,
                socioeconomic and mobility predictors.
            </p>
            <div class="metric-label">Operational & temporal</div>
            <div>{chips(["Direction", "Operating period", "Headway", "Relative stop position"])}</div>
            <br>
            <div class="metric-label">Built environment · 400 m</div>
            <div>{chips(["Employment", "Retail", "Education", "Healthcare"])}</div>
            <br>
            <div class="metric-label">Population & demographics</div>
            <div>{chips(["Population", "Student rate", "Age 0–14", "Age 65+", "Female rate"])}</div>
            <br>
            <div class="metric-label">Socioeconomic & mobility</div>
            <div>{chips(["Deprivation domains", "No-car households", "Working from home"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="spec-card-health">
            <h3 style="margin-top:0">Health-Integrated Model</h3>
            <p style="color:{GREY};font-size:.86rem;">
                The complete Baseline model plus three LSOA-level health and disability indicators.
            </p>
            <div style="background:#F0F3FF;padding:.7rem;border-radius:.55rem;">
                <strong>Baseline feature set</strong><br>
                <span style="font-size:.78rem;color:{GREY};">All conventional predictors retained</span>
            </div>
            <div style="font-size:1.7rem;font-weight:800;text-align:center;margin:.45rem 0;">+</div>
            <div style="background:#FFF9D8;padding:.8rem;border-radius:.55rem;">
                <div class="metric-label">Added health block</div>
                <div>{chips(["Poor general health", "Activity-limiting disability", "Health deprivation score"])}</div>
            </div>
            <p style="font-size:.77rem;color:{GREY};margin-top:.9rem;">
                These variables describe neighbourhood context, not individual passenger health.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="insight">
    <strong>Controlled comparison.</strong> Both specifications use the same observations,
    grouped train/test partition, cross-validation folds, preprocessing, classifier family,
    tuning framework and random seed. The comparison therefore isolates the incremental
    predictive contribution of health information.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# =============================================================================
# FOUR ANALYTICAL TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Model Selection",
        "2. Test Performance",
        "3. Health Incremental Value",
        "4. Model Interpretation",
    ]
)


# =============================================================================
# TAB 1 — MODEL SELECTION
# =============================================================================
with tab1:
    st.subheader("Development-Set Model Selection")
    st.caption(
        "Four classifier families evaluated using five-fold cross-validation grouped by trip."
    )

    metric = st.radio(
        "Evaluation metric",
        [
            "Log Loss",
            "Brier Score",
            "ROC-AUC",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "Specificity",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="cv_metric",
    )

    lower_better = metric in {"Log Loss", "Brier Score"}
    sd_col = metric + " SD"

    plot_df = cv_df[["Model", metric, sd_col]].copy()

    colors = [
        BEE_YELLOW if m == "XGBoost" else ("#5F5E5E" if m == "Random Forest" else "#C8C6C5")
        for m in plot_df["Model"]
    ]

    fig = go.Figure(
        go.Bar(
            x=plot_df[metric],
            y=plot_df["Model"],
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data",
                array=plot_df[sd_col],
                visible=True,
                color="#7E8794",
                thickness=1,
            ),
            text=[f"{v:.4f}" for v in plot_df[metric]],
            textposition="outside",
            hovertemplate="%{y}<br>" + metric + ": %{x:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"{metric} comparison — {'lower' if lower_better else 'higher'} is better",
        xaxis_title=metric,
        yaxis_title=None,
        showlegend=False,
    )
    standard_plot_layout(fig, height=390)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    display_cols = [
        "Model",
        "Log Loss",
        "Brier Score",
        "ROC-AUC",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Specificity",
    ]
    table = cv_df[display_cols].copy()

    def highlight_xgb(row):
        return [
            "background-color: #FFF5B5; font-weight: 700;" if row["Model"] == "XGBoost" else ""
            for _ in row
        ]

    st.dataframe(
        table.style.apply(highlight_xgb, axis=1).format(
            {c: "{:.4f}" for c in display_cols if c != "Model"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="insight">
        <strong>Why XGBoost?</strong> Random Forest achieved marginally higher ROC-AUC,
        balanced accuracy, precision and specificity. XGBoost nevertheless produced the
        <strong>lowest log loss and Brier score</strong>. Because log loss was the
        pre-specified selection criterion, XGBoost was retained while maintaining
        discrimination comparable with Random Forest.
        </div>
        """,
        unsafe_allow_html=True,
    )



# =============================================================================
# TAB 2 — TEST PERFORMANCE
# =============================================================================
with tab2:
    st.subheader("Generalisation to Unseen Trips")
    st.caption("Final XGBoost performance on the untouched grouped test set.")

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Log Loss", f"{test_result['Log Loss']:.4f}", "Probabilistic performance")
    with b:
        metric_card("ROC-AUC", f"{test_result['ROC-AUC']:.4f}", "Ranking discrimination")
    with c:
        metric_card("Brier Score", f"{test_result['Brier Score']:.4f}", "Probability error")
    with d:
        metric_card("F1 Score", f"{test_result['F1']:.4f}", "Precision–recall balance")

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Accuracy", f"{test_result['Accuracy']:.2%}", "Threshold = 0.50")
    with b:
        metric_card("Precision", f"{test_result['Precision']:.2%}", "Positive predictive value")
    with c:
        metric_card("Recall", f"{test_result['Recall']:.2%}", "Sensitivity to boarding occurrence")
    with d:
        metric_card("Specificity", f"{test_result['Specificity']:.2%}", "No-boarding recognition")

    st.markdown(
        """
        <div class="warning-note">
        <strong>Error profile.</strong> At the 0.50 threshold, recall is substantially higher
        than specificity. The model therefore detects most positive boarding-occurrence
        visits, but produces more false positives among no-demand visits.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("#### Confusion matrix · threshold 0.50")
        cm_z = np.array(
            [
                [test_result["TN"], test_result["FP"]],
                [test_result["FN"], test_result["TP"]],
            ]
        )
        cm_text = np.array(
            [
                [f"TN<br>{test_result['TN']:,}", f"FP<br>{test_result['FP']:,}"],
                [f"FN<br>{test_result['FN']:,}", f"TP<br>{test_result['TP']:,}"],
            ]
        )
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm_z,
                x=["Predicted: No boarding", "Predicted: Boarding"],
                y=["Actual: No boarding", "Actual: Boarding"],
                text=cm_text,
                texttemplate="%{text}",
                textfont={"size": 16, "family": "Public Sans"},
                colorscale=[
                    [0.0, "#F0F3FF"],
                    [0.55, "#DCE2F3"],
                    [1.0, "#FFD800"],
                ],
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
            )
        )
        standard_plot_layout(cm_fig, height=380)
        cm_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(cm_fig, use_container_width=True, config={"displaylogo": False})

    with c2:
        st.markdown("#### ROC curve")
        if (
            pred_df is not None
            and pred_y_col
            and pred_p_col
            and roc_curve is not None
        ):
            y_true = pred_df[pred_y_col].astype(int).to_numpy()
            prob = pred_df[pred_p_col].to_numpy()
            fpr, tpr, _ = roc_curve(y_true, prob)
            auc = roc_auc_score(y_true, prob)

            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"XGBoost · AUC {auc:.4f}",
                    line=dict(color=BLUE, width=3),
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Chance",
                    line=dict(color="#9CA3AF", width=1.5, dash="dash"),
                )
            )
            roc_fig.update_layout(
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(orientation="h", y=1.08),
            )
            roc_fig.update_xaxes(range=[0, 1])
            roc_fig.update_yaxes(range=[0, 1])
            standard_plot_layout(roc_fig, height=380)
            st.plotly_chart(roc_fig, use_container_width=True, config={"displaylogo": False})
        else:
            st.info(
                "ROC curve requires row-level actual labels and predicted probabilities in "
                "`data/model_outputs/test_predictions.csv`. The dissertation summary metrics "
                "above remain available."
            )

    st.markdown("#### Decision Threshold Explorer")
    if (
        pred_df is not None
        and pred_y_col
        and pred_p_col
        and confusion_matrix is not None
    ):
        threshold = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.01,
            key="threshold",
        )
        dyn = compute_threshold_metrics(
            pred_df[pred_y_col].astype(int).to_numpy(),
            pred_df[pred_p_col].to_numpy(),
            threshold,
        )

        x1, x2, x3, x4 = st.columns(4)
        with x1:
            metric_card("Precision", f"{dyn['Precision']:.2%}", f"threshold = {threshold:.2f}")
        with x2:
            metric_card("Recall", f"{dyn['Recall']:.2%}", f"threshold = {threshold:.2f}")
        with x3:
            metric_card("Specificity", f"{dyn['Specificity']:.2%}", f"threshold = {threshold:.2f}")
        with x4:
            metric_card(
                "Balanced Accuracy",
                f"{dyn['Balanced Accuracy']:.2%}",
                f"threshold = {threshold:.2f}",
            )

        st.caption(
            "The explorer recomputes threshold-dependent metrics directly from "
            "`test_predictions.csv`. ROC-AUC, log loss and Brier score do not change with "
            "the classification cut-off."
        )
    else:
        st.info(
            "The threshold explorer is intentionally disabled because the app could not "
            "identify both the actual-label and predicted-probability columns in "
            "`test_predictions.csv`. No approximate threshold curve is fabricated."
        )



# =============================================================================
# TAB 3 — HEALTH INCREMENTAL VALUE
# =============================================================================
with tab3:
    st.subheader("Does Health Improve Prediction?")
    st.caption(
        "Direct Baseline versus Health-Integrated comparison under the same modelling framework."
    )

    view = st.radio(
        "Evaluation view",
        ["Untouched Test", "Cross-Validation"],
        horizontal=True,
        label_visibility="collapsed",
        key="incremental_view",
    )

    comp_df, comp_source = load_comparison(view)

    test_comp = FINAL_COMPARE["Untouched Test"].copy()
    test_comp["Delta"] = test_comp["Health-Integrated"] - test_comp["Baseline"]
    delta_log = float(
        test_comp.loc[test_comp["Metric"] == "Log Loss", "Delta"].iloc[0]
    )
    delta_auc = float(
        test_comp.loc[test_comp["Metric"] == "ROC-AUC", "Delta"].iloc[0]
    )

    st.markdown(
        f"""
        <div class="verdict">
            <div class="metric-label" style="color:#544600;">Empirical RQ1 verdict</div>
            <h3>LIMITED INCREMENTAL PREDICTIVE VALUE</h3>
            <div style="font-size:.9rem;">
                On the untouched test set, adding health information changed log loss by
                <strong>{delta_log:+.5f}</strong> and ROC-AUC by
                <strong>{delta_auc:+.5f}</strong>. The changes are practically very small,
                so the Baseline model is preferred for prediction on grounds of parsimony.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Paired plot. A common 0–1 scale deliberately avoids visually exaggerating tiny differences.
    y_order = list(comp_df["Metric"])[::-1]
    dumbbell = go.Figure()

    for _, row in comp_df.iterrows():
        dumbbell.add_trace(
            go.Scatter(
                x=[row["Baseline"], row["Health-Integrated"]],
                y=[row["Metric"], row["Metric"]],
                mode="lines",
                line=dict(color="#D1D5DB", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    dumbbell.add_trace(
        go.Scatter(
            x=comp_df["Baseline"],
            y=comp_df["Metric"],
            mode="markers",
            name="Baseline",
            marker=dict(size=11, color="#5F5E5E"),
            hovertemplate="%{y}<br>Baseline: %{x:.5f}<extra></extra>",
        )
    )
    dumbbell.add_trace(
        go.Scatter(
            x=comp_df["Health-Integrated"],
            y=comp_df["Metric"],
            mode="markers",
            name="Health-Integrated",
            marker=dict(size=12, color=BEE_YELLOW, line=dict(color=BLACK, width=1)),
            hovertemplate="%{y}<br>Health-Integrated: %{x:.5f}<extra></extra>",
        )
    )
    dumbbell.update_layout(
        xaxis_title="Metric value",
        yaxis_title=None,
        legend=dict(orientation="h", y=1.10),
    )
    dumbbell.update_yaxes(categoryorder="array", categoryarray=y_order)
    dumbbell.update_xaxes(range=[0, 0.90])
    standard_plot_layout(dumbbell, height=430)
    st.plotly_chart(dumbbell, use_container_width=True, config={"displaylogo": False})

    show = comp_df.copy()
    show["Delta"] = show["Health-Integrated"] - show["Baseline"]
    st.dataframe(
        show.style.format(
            {
                "Baseline": "{:.5f}",
                "Health-Integrated": "{:.5f}",
                "Delta": "{:+.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(
            """
            <div class="insight">
            <strong>Prediction.</strong><br>
            Health and disability add little information beyond the conventional feature
            set for predicting reconstructed boarding occurrence.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="insight">
            <strong>Interpretation.</strong><br>
            Limited incremental predictive value does not make health irrelevant. Health
            remains useful as an equity-sensitive layer for interpreting where apparently
            low realised use coincides with greater vulnerability.
            </div>
            """,
            unsafe_allow_html=True,
        )



# =============================================================================
# TAB 4 — MODEL INTERPRETATION
# =============================================================================
with tab4:
    st.subheader("What Drives XGBoost Predictions?")
    st.caption(
        "Global SHAP importance is displayed from the model-output exports. Values are "
        "model-specific predictive contributions, not causal effects."
    )

    specification = st.radio(
        "Model specification",
        ["Baseline", "Health-Integrated"],
        horizontal=True,
        label_visibility="collapsed",
        key="shap_spec",
    )

    shap_path = SHAP_BASE_PATH if specification == "Baseline" else SHAP_HEALTH_PATH
    shap_df = load_shap(shap_path)

    if shap_df is None or shap_df.empty:
        st.warning(
            f"Could not parse SHAP values from `{shap_path.name}`. "
            "The rest of RQ1 is still available, but this chart is deliberately not "
            "fabricated. Expected either (a) `feature` + importance columns, or "
            "(b) observation-level numeric SHAP columns."
        )
    else:
        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=min(25, len(shap_df)),
            value=min(12, len(shap_df)),
            step=1,
            key=f"topn_{specification}",
        )
        top = shap_df.head(top_n).copy()
        top["Domain"] = top["Feature"].map(feature_domain)
        top = top.sort_values("MeanAbsSHAP", ascending=True)

        shap_fig = go.Figure(
            go.Bar(
                x=top["MeanAbsSHAP"],
                y=top["Feature"],
                orientation="h",
                marker_color=[DOMAIN_COLOURS[d] for d in top["Domain"]],
                customdata=top[["Domain"]],
                hovertemplate=(
                    "%{y}<br>Mean |SHAP|: %{x:.5f}<br>"
                    "Domain: %{customdata[0]}<extra></extra>"
                ),
            )
        )
        shap_fig.update_layout(
            xaxis_title="Mean |SHAP|",
            yaxis_title=None,
            title=f"Global feature importance · {specification}",
        )
        standard_plot_layout(shap_fig, height=max(400, 28 * top_n + 120))
        st.plotly_chart(shap_fig, use_container_width=True, config={"displaylogo": False})

        inspect = st.selectbox(
            "Inspect feature",
            options=list(shap_df["Feature"]),
            key=f"inspect_{specification}",
        )
        row = shap_df.reset_index(drop=True)
        selected_idx = int(row.index[row["Feature"] == inspect][0])
        selected_val = float(row.loc[selected_idx, "MeanAbsSHAP"])
        domain = feature_domain(inspect)

        q1, q2, q3 = st.columns(3)
        with q1:
            metric_card("Feature", inspect, domain)
        with q2:
            metric_card("Global rank", f"#{selected_idx + 1}", f"of {len(row)} parsed features")
        with q3:
            metric_card("Mean |SHAP|", f"{selected_val:.5f}", "Global model contribution")

        if specification == "Health-Integrated":
            health_mask = shap_df["Feature"].astype(str).str.lower().str.contains(
                "health|disab|limiting", regex=True
            )
            health_features = shap_df.loc[health_mask].copy()
            if not health_features.empty:
                st.markdown("#### Health-feature placement")
                ranks = shap_df.reset_index(drop=True)
                ranks["Rank"] = ranks.index + 1
                health_rank = ranks[
                    ranks["Feature"].astype(str).str.lower().str.contains(
                        "health|disab|limiting", regex=True
                    )
                ][["Rank", "Feature", "MeanAbsSHAP"]]
                st.dataframe(
                    health_rank.style.format({"MeanAbsSHAP": "{:.5f}"}),
                    hide_index=True,
                    use_container_width=True,
                )

        st.markdown(
            """
            <div class="warning-note">
            <strong>Interpretation caveat.</strong> SHAP values describe model-specific
            predictive contribution. They should not be interpreted as independent causal
            effects because operational, demographic, deprivation and health variables are
            correlated. In addition, the reconstructed target is structurally related to
            some operational inputs.
            </div>
            """,
            unsafe_allow_html=True,
        )



# =============================================================================
# RQ1 SYNTHESIS
# =============================================================================
st.divider()
st.subheader("RQ1 Answer")

r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(
        """
        <div class="method-card">
        <div class="metric-label">1 · Predictive foundation</div>
        <h3>Conventional predictors perform well</h3>
        <div class="metric-sub">
        Operational, temporal, built-environment, demographic, socioeconomic and
        mobility information provides the main predictive foundation.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with r2:
    st.markdown(
        """
        <div class="method-card-yellow">
        <div class="metric-label">2 · Incremental test</div>
        <h3>Health adds little predictive gain</h3>
        <div class="metric-sub">
        Baseline and Health-Integrated out-of-sample metrics are nearly identical.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with r3:
    st.markdown(
        """
        <div class="method-card">
        <div class="metric-label">3 · Planning role</div>
        <h3>Health remains interpretively useful</h3>
        <div class="metric-sub">
        Health is retained as an equity-sensitive contextual layer rather than as a
        necessary source of predictive complexity.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="insight" style="margin-top:1rem;">
    <strong>Formal RQ1 synthesis.</strong>
    Health and disability indicators provide limited incremental value for predicting
    reconstructed boarding occurrence once conventional demand determinants are represented.
    Their greater contribution therefore lies in interpreting realised demand rather than
    improving prediction alone.
    </div>
    """,
    unsafe_allow_html=True,
)

if hasattr(st, "page_link"):
    try:
        st.page_link(
            "pages/3_Equity_Analysis.py",
            label="Continue to Equity Analysis (RQ2) →",
            icon="➡️",
        )
    except Exception:
        pass
