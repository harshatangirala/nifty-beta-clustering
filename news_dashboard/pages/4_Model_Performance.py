"""
Page 4 – NLP Model Performance Dashboard

Evaluates FinBERT and VADER on Financial PhraseBank (gold-standard benchmark).
Metrics: Accuracy, Precision, Recall, F1 (per-class + macro + weighted),
         MCC, Cohen's Kappa, ROC-AUC, Average Precision, Confusion Matrix,
         Reliability / Calibration diagram, live inference demo.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.metric-card { background:#1A1D2E; border-radius:12px; padding:16px 20px;
               border:1px solid #2D3153; text-align:center; }
.metric-card h2 { font-size:2rem; margin:4px 0; }
.metric-card p  { font-size:.82rem; color:#9AA0B4; margin:0; }
.model-badge-finbert { background:#1B3A5C; color:#4ECDC4; border-radius:8px;
                       padding:4px 12px; font-weight:700; font-size:.82rem; }
.model-badge-vader   { background:#3A2A1B; color:#F39C12; border-radius:8px;
                       padding:4px 12px; font-weight:700; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_analyzer():
    from src.sentiment_analyzer import SentimentAnalyzer
    from config import HF_TOKEN
    return SentimentAnalyzer(hf_token=HF_TOKEN)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<h2>🤖 NLP Sentiment Model Performance</h2>
<p style='color:#9AA0B4;'>
  Evaluation on the <strong>Financial PhraseBank</strong> benchmark
  (Malo et al., 2014) — gold-standard financial NLP dataset.
  Comparing <span class='model-badge-finbert'>FinBERT</span>
  vs <span class='model-badge-vader'>VADER</span>
</p>
""", unsafe_allow_html=True)

# ── Model info ────────────────────────────────────────────────────────────────

st.subheader("📐 Model Architecture Overview")
col_fb, col_vd = st.columns(2)

with col_fb:
    st.markdown("""
<div class='metric-card' style='text-align:left;'>
  <span class='model-badge-finbert'>FinBERT</span><br><br>
  <b>Base:</b> BERT-base-uncased (110M parameters)<br>
  <b>Fine-tuning:</b> ProsusAI/finbert on 10,000+ financial news articles<br>
  <b>Task:</b> 3-class text classification (positive / negative / neutral)<br>
  <b>Input:</b> Max 512 tokens, truncated<br>
  <b>Output:</b> Softmax probabilities per class<br>
  <b>Inference:</b> HuggingFace Inference API (no local GPU/download needed)<br>
  <b>Strength:</b> Financial domain expertise, contextual BERT embeddings
</div>
""", unsafe_allow_html=True)

with col_vd:
    st.markdown("""
<div class='metric-card' style='text-align:left;'>
  <span class='model-badge-vader'>VADER</span><br><br>
  <b>Type:</b> Rule-based lexicon + heuristics (0 parameters)<br>
  <b>Lexicon:</b> 7,500+ lexical features with valence scores<br>
  <b>Task:</b> Compound score → positive / negative / neutral<br>
  <b>Input:</b> Any text length<br>
  <b>Output:</b> pos / neg / neu / compound scores<br>
  <b>Strength:</b> No training needed, fast, interpretable<br>
  <b>Speed:</b> ~10,000 articles/sec
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Evaluation ────────────────────────────────────────────────────────────────

st.subheader("📊 Benchmark Evaluation")
st.info(
    "Evaluation uses **Financial PhraseBank** (sentences_allagree split) — "
    "4,845 sentences from financial news, labelled by domain experts. "
    "We use up to 500 samples for the demo (or the embedded 30-sample fallback if offline)."
)

n_eval = st.slider("Number of evaluation samples", 30, 500, 200, 10)

eval_btn = st.button(
    "▶️  Run Evaluation", type="primary",
    help="This may take 1–5 minutes on CPU (FinBERT inference).",
)

@st.cache_data(show_spinner="Running evaluation…", ttl=3600)
def run_evaluation(n: int):
    from src.sentiment_analyzer import ModelEvaluator
    analyzer = get_analyzer()
    evaluator = ModelEvaluator(analyzer)
    return evaluator.evaluate(n_samples=n)

if eval_btn or st.session_state.get("eval_done"):
    with st.spinner("Running FinBERT + VADER on Financial PhraseBank…"):
        results = run_evaluation(n_eval)
    st.session_state["eval_done"]    = True
    st.session_state["eval_results"] = results

if "eval_results" not in st.session_state:
    st.warning("Click **Run Evaluation** to compute metrics.")
    st.stop()

results = st.session_state["eval_results"]
fb      = results["finbert"]
vd      = results["vader"]
labels  = ["positive", "negative", "neutral"]
COLORS  = {"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"}
MODEL_COLORS = {"FinBERT": "#4ECDC4", "VADER": "#F39C12"}

st.success(
    f"✅ Evaluated {fb['n_samples']} samples from Financial PhraseBank. "
    f"FinBERT Accuracy: **{fb['accuracy']:.1%}** | VADER Accuracy: **{vd['accuracy']:.1%}**"
)
st.markdown("---")

# ── KPI comparison ────────────────────────────────────────────────────────────

st.subheader("🏆 Head-to-Head Comparison")

def kpi_row(metric, fb_val, vd_val, fmt=".3f", higher_better=True):
    fb_v = fb_val if fb_val is not None else float("nan")
    vd_v = vd_val if vd_val is not None else float("nan")
    fb_fmt = f"{fb_v:{fmt}}" if not np.isnan(fb_v) else "N/A"
    vd_fmt = f"{vd_v:{fmt}}" if not np.isnan(vd_v) else "N/A"
    winner = ""
    if not np.isnan(fb_v) and not np.isnan(vd_v):
        if (higher_better and fb_v > vd_v) or (not higher_better and fb_v < vd_v):
            winner = "🏅 FinBERT"
        else:
            winner = "🏅 VADER"
    return {"Metric": metric, "FinBERT": fb_fmt, "VADER": vd_fmt, "Winner": winner}

rows = [
    kpi_row("Accuracy",           fb["accuracy"],        vd["accuracy"],        ".1%"),
    kpi_row("Macro F1",            fb["macro_f1"],         vd["macro_f1"],         ".4f"),
    kpi_row("Macro Precision",     fb["macro_precision"],  vd["macro_precision"],  ".4f"),
    kpi_row("Macro Recall",        fb["macro_recall"],     vd["macro_recall"],     ".4f"),
    kpi_row("Weighted F1",         fb["weighted_f1"],      vd["weighted_f1"],      ".4f"),
    kpi_row("MCC",                 fb["mcc"],              vd["mcc"],              ".4f"),
    kpi_row("Cohen's Kappa",       fb["kappa"],            vd["kappa"],            ".4f"),
    kpi_row("ROC-AUC (macro)",     fb.get("roc_auc_macro"), vd.get("roc_auc_macro"), ".4f"),
    kpi_row("Avg Precision (macro)",fb.get("avg_precision"), vd.get("avg_precision"), ".4f"),
]
cmp_df = pd.DataFrame(rows)
st.dataframe(
    cmp_df.style.apply(
        lambda col: [
            "color: #4ECDC4; font-weight:bold" if v == "🏅 FinBERT"
            else ("color: #F39C12; font-weight:bold" if v == "🏅 VADER" else "")
            for v in col
        ] if col.name == "Winner" else [""] * len(col),
        axis=0,
    ),
    use_container_width=True,
    height=380,
)

st.markdown("---")

# ── Per-class metrics ─────────────────────────────────────────────────────────

st.subheader("📋 Per-Class Metrics")

tab_fb, tab_vd = st.tabs(["FinBERT", "VADER"])

def per_class_table(metrics_dict):
    pc = metrics_dict["per_class"]
    rows_pc = []
    for cls in labels:
        if cls in pc:
            rows_pc.append({
                "Class":     cls.capitalize(),
                "Precision": f"{pc[cls]['precision']:.4f}",
                "Recall":    f"{pc[cls]['recall']:.4f}",
                "F1-Score":  f"{pc[cls]['f1']:.4f}",
                "Support":   int(pc[cls]["support"]),
            })
    return pd.DataFrame(rows_pc)

def per_class_radar(metrics_dict, title, color):
    pc = metrics_dict["per_class"]
    cats = ["Precision", "Recall", "F1-Score"]
    fig = go.Figure()
    for cls in labels:
        if cls not in pc:
            continue
        vals = [pc[cls]["precision"], pc[cls]["recall"], pc[cls]["f1"]]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            name=cls.capitalize(),
            line=dict(color=COLORS[cls]),
            fill="toself", opacity=0.4,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1A1D2E",
            radialaxis=dict(visible=True, range=[0, 1], color="#9AA0B4", gridcolor="#2D3153"),
            angularaxis=dict(color="#E8EAED"),
        ),
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"),
        showlegend=True, height=320,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=30, r=30, t=30, b=30),
        title=dict(text=title, font=dict(color="#E8EAED")),
    )
    return fig

with tab_fb:
    col_t, col_r = st.columns(2)
    with col_t:
        st.dataframe(per_class_table(fb), use_container_width=True, height=175)
        # Per-class bar chart
        pc_data = []
        for cls in labels:
            if cls in fb["per_class"]:
                pc_data.append({"Class": cls.capitalize(), "Metric": "Precision",
                                "Value": fb["per_class"][cls]["precision"]})
                pc_data.append({"Class": cls.capitalize(), "Metric": "Recall",
                                "Value": fb["per_class"][cls]["recall"]})
                pc_data.append({"Class": cls.capitalize(), "Metric": "F1",
                                "Value": fb["per_class"][cls]["f1"]})
        pc_df = pd.DataFrame(pc_data)
        fig_pc = px.bar(
            pc_df, x="Class", y="Value", color="Metric", barmode="group",
            color_discrete_sequence=["#4ECDC4", "#FF6B6B", "#F39C12"],
            labels={"Value": "Score"},
        )
        fig_pc.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=260,
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(range=[0, 1], gridcolor="#2D3153"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pc, use_container_width=True)
    with col_r:
        st.plotly_chart(per_class_radar(fb, "FinBERT Per-Class Radar", "#4ECDC4"),
                        use_container_width=True)

with tab_vd:
    col_t2, col_r2 = st.columns(2)
    with col_t2:
        st.dataframe(per_class_table(vd), use_container_width=True, height=175)
        pc_data2 = []
        for cls in labels:
            if cls in vd["per_class"]:
                pc_data2.append({"Class": cls.capitalize(), "Metric": "Precision",
                                 "Value": vd["per_class"][cls]["precision"]})
                pc_data2.append({"Class": cls.capitalize(), "Metric": "Recall",
                                 "Value": vd["per_class"][cls]["recall"]})
                pc_data2.append({"Class": cls.capitalize(), "Metric": "F1",
                                 "Value": vd["per_class"][cls]["f1"]})
        pc_df2 = pd.DataFrame(pc_data2)
        fig_pc2 = px.bar(
            pc_df2, x="Class", y="Value", color="Metric", barmode="group",
            color_discrete_sequence=["#F39C12", "#FF6B6B", "#4ECDC4"],
            labels={"Value": "Score"},
        )
        fig_pc2.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=260,
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(range=[0, 1], gridcolor="#2D3153"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pc2, use_container_width=True)
    with col_r2:
        st.plotly_chart(per_class_radar(vd, "VADER Per-Class Radar", "#F39C12"),
                        use_container_width=True)

st.markdown("---")

# ── Confusion matrices ────────────────────────────────────────────────────────

st.subheader("🔲 Confusion Matrices")

def plot_cm(cm_list, title, color):
    cm = np.array(cm_list)
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    text = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)"
             for j in range(3)] for i in range(3)]
    cls_labels = ["Positive", "Negative", "Neutral"]
    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f"Pred: {l}" for l in cls_labels],
        y=[f"True: {l}" for l in cls_labels],
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#1A1D2E"], [1, color]],
        showscale=False,
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"),
        title=dict(text=title, font=dict(color="#E8EAED", size=14)),
        height=300,
        xaxis=dict(side="bottom"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

col_cm1, col_cm2 = st.columns(2)
with col_cm1:
    st.plotly_chart(plot_cm(fb["confusion_matrix"], "FinBERT Confusion Matrix", "#4ECDC4"),
                    use_container_width=True)
with col_cm2:
    st.plotly_chart(plot_cm(vd["confusion_matrix"], "VADER Confusion Matrix", "#F39C12"),
                    use_container_width=True)

st.markdown("---")

# ── ROC curves (FinBERT only) ─────────────────────────────────────────────────

st.subheader("📈 ROC Curves (One-vs-Rest)")

if fb.get("probs") is not None:
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_true = fb["true_labels"]
        y_probs = np.array(fb["probs"])
        y_bin   = label_binarize(y_true, classes=labels)

        fig_roc = go.Figure()
        for i, cls in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{cls.capitalize()} (AUC={roc_auc_val:.3f})",
                line=dict(color=COLORS[cls], width=2),
                mode="lines",
            ))

        # VADER pseudo-ROC using compound score mapped to binary
        vd_compound = []
        for t in fb["true_labels"]:
            vd_r = get_analyzer().vader.analyze(
                next((s for s, l in zip(fb.get("true_labels", []), fb.get("pred_labels", [])) if l == t), t)
            )
            vd_compound.append(vd_r.get("compound", 0) if vd_r else 0)

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Classifier",
            line=dict(color="#9AA0B4", dash="dash", width=1),
            mode="lines",
        ))
        fig_roc.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=380,
            xaxis=dict(title="False Positive Rate", gridcolor="#2D3153", range=[0, 1]),
            yaxis=dict(title="True Positive Rate",  gridcolor="#2D3153", range=[0, 1.02]),
            legend=dict(x=0.55, y=0.05, bgcolor="#1A1D2E", bordercolor="#2D3153"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception as e:
        st.warning(f"ROC curve computation skipped: {e}")

st.markdown("---")

# ── Calibration / Reliability diagram ─────────────────────────────────────────

st.subheader("📏 Confidence Calibration (FinBERT)")
st.caption(
    "A well-calibrated model has predicted probabilities matching observed frequencies. "
    "Points near the diagonal = good calibration."
)

if fb.get("probs") is not None:
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(fb["true_labels"], classes=labels)
        y_probs    = np.array(fb["probs"])

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Perfect Calibration",
            line=dict(color="#9AA0B4", dash="dash"), mode="lines",
        ))
        for i, cls in enumerate(labels):
            try:
                prob_true, prob_pred = calibration_curve(
                    y_true_bin[:, i], y_probs[:, i], n_bins=8
                )
                fig_cal.add_trace(go.Scatter(
                    x=prob_pred, y=prob_true,
                    name=f"{cls.capitalize()}",
                    line=dict(color=COLORS[cls], width=2),
                    mode="lines+markers",
                ))
            except Exception:
                pass

        fig_cal.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=340,
            xaxis=dict(title="Mean Predicted Probability", gridcolor="#2D3153", range=[0, 1]),
            yaxis=dict(title="Fraction of Positives",      gridcolor="#2D3153", range=[0, 1]),
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_cal, use_container_width=True)
    except Exception as e:
        st.warning(f"Calibration curve skipped: {e}")

st.markdown("---")

# ── Error analysis ────────────────────────────────────────────────────────────

st.subheader("🔎 Error Analysis — Prediction Samples")

tab_correct, tab_wrong = st.tabs(["✅ Correct Predictions", "❌ Wrong Predictions"])

def sample_rows(true_lbls, pred_lbls, texts_list, correct: bool, n: int = 10):
    rows = []
    for t, p, tx in zip(true_lbls, pred_lbls, texts_list):
        if (t == p) == correct:
            rows.append({"Text": tx[:180], "True": t, "Predicted": p})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)

# We need to get the original texts for the display
texts_for_display, _ = (
    get_analyzer().finbert.analyze.__self__.__class__
    if False else (None, None)
), None

if fb.get("true_labels") and fb.get("pred_labels"):
    # Get sample texts from the evaluator's phrasebank
    try:
        from src.sentiment_analyzer import ModelEvaluator
        evaluator = ModelEvaluator(get_analyzer())
        sample_texts, _ = evaluator.load_phrasebank(n=len(fb["true_labels"]))
    except Exception:
        sample_texts = [f"[Article {i+1}]" for i in range(len(fb["true_labels"]))]

    with tab_correct:
        correct_df = sample_rows(fb["true_labels"], fb["pred_labels"], sample_texts, True)
        if not correct_df.empty:
            st.dataframe(
                correct_df.style.applymap(
                    lambda v: f"color: {COLORS.get(v, '#E8EAED')}", subset=["True", "Predicted"]
                ),
                use_container_width=True, height=330,
            )

    with tab_wrong:
        wrong_df = sample_rows(fb["true_labels"], fb["pred_labels"], sample_texts, False)
        if not wrong_df.empty:
            st.dataframe(
                wrong_df.style.applymap(
                    lambda v: f"color: {COLORS.get(v, '#E8EAED')}", subset=["True", "Predicted"]
                ),
                use_container_width=True, height=330,
            )

st.markdown("---")

# ── Live Inference Demo ────────────────────────────────────────────────────────

st.subheader("🎯 Live Sentiment Inference Demo")

demo_text = st.text_area(
    "Enter any financial news headline or sentence:",
    value="Infosys reports record quarterly revenue of $4.9 billion, beating analyst expectations by 8%.",
    height=100,
)

if st.button("🔍 Analyse Sentiment", type="primary"):
    analyzer = get_analyzer()
    with st.spinner("Running FinBERT + VADER…"):
        fb_result = analyzer.finbert.analyze(demo_text)
        vd_result = analyzer.vader.analyze(demo_text)

    col_fb2, col_vd2 = st.columns(2)

    with col_fb2:
        st.markdown("**FinBERT Prediction**")
        if fb_result:
            lbl   = fb_result["label"]
            color = {"positive": "#2ECC71", "negative": "#E74C3C"}.get(lbl, "#95A5A6")
            st.markdown(
                f"<h3 style='color:{color}'>{'📈' if lbl=='positive' else '📉' if lbl=='negative' else '➡️'} "
                f"{lbl.capitalize()}</h3>",
                unsafe_allow_html=True,
            )
            prob_df = pd.DataFrame({
                "Class": ["Positive", "Negative", "Neutral"],
                "Probability": [fb_result["positive"], fb_result["negative"], fb_result["neutral"]],
            })
            fig_prob = px.bar(
                prob_df, x="Class", y="Probability",
                color="Class",
                color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C", "Neutral": "#95A5A6"},
                range_y=[0, 1],
            )
            fig_prob.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font=dict(color="#E8EAED"), showlegend=False, height=220,
                yaxis=dict(gridcolor="#2D3153"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            st.metric("Compound Score", f"{fb_result['compound']:+.4f}")
        else:
            st.warning("FinBERT not available (install PyTorch + Transformers)")

    with col_vd2:
        st.markdown("**VADER Prediction**")
        if vd_result:
            lbl   = vd_result["label"]
            color = {"positive": "#2ECC71", "negative": "#E74C3C"}.get(lbl, "#95A5A6")
            st.markdown(
                f"<h3 style='color:{color}'>{'📈' if lbl=='positive' else '📉' if lbl=='negative' else '➡️'} "
                f"{lbl.capitalize()}</h3>",
                unsafe_allow_html=True,
            )
            vd_df = pd.DataFrame({
                "Class": ["Positive", "Negative", "Neutral"],
                "Score": [vd_result["positive"], vd_result["negative"], vd_result["neutral"]],
            })
            fig_vd = px.bar(
                vd_df, x="Class", y="Score",
                color="Class",
                color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C", "Neutral": "#95A5A6"},
                range_y=[0, 1],
            )
            fig_vd.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font=dict(color="#E8EAED"), showlegend=False, height=220,
                yaxis=dict(gridcolor="#2D3153"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_vd, use_container_width=True)
            st.metric("Compound Score", f"{vd_result['compound']:+.4f}")
        else:
            st.warning("VADER not available (pip install vaderSentiment)")

# ── Metric explanations ───────────────────────────────────────────────────────

with st.expander("📖 Metric Definitions"):
    st.markdown("""
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | % of all predictions correct |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are truly positive |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many were detected |
| **F1-Score** | 2·P·R / (P+R) | Harmonic mean of Precision & Recall |
| **Macro F1** | Mean F1 across all classes (unweighted) | Overall class balance |
| **Weighted F1** | F1 weighted by class support | Accounts for class imbalance |
| **MCC** | (TP·TN − FP·FN) / √ … | Most balanced single metric for multi-class |
| **Cohen's Kappa** | (Po − Pe) / (1 − Pe) | Agreement beyond chance |
| **ROC-AUC** | Area under ROC curve (OvR macro) | Discrimination ability across thresholds |
| **Avg Precision** | Area under Precision-Recall curve | Better than AUC for imbalanced classes |

**Why FinBERT outperforms VADER for financial text:**
- BERT uses contextual embeddings; the same word can have different sentiment based on surrounding context
- FinBERT was specifically fine-tuned on financial news, learning domain-specific sentiment patterns
- VADER treats financial jargon as neutral; FinBERT understands that "headwinds" is bearish
""")
