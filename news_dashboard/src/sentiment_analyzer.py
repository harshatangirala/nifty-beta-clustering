"""
Sentiment analysis using FinBERT (ProsusAI/finbert) with VADER fallback.

FinBERT is a BERT model fine-tuned on 10,000+ financial news sentences.
Label order:  index 0 → positive  |  index 1 → negative  |  index 2 → neutral
"""

from __future__ import annotations
import re
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Text helpers ─────────────────────────────────────────────────────────────

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")

def _clean(text: str) -> str:
    text = _HTML_RE.sub(" ", text or "")
    text = _WS_RE.sub(" ", text).strip()
    return text[:1024]  # hard cap


# ── FinBERT wrapper ──────────────────────────────────────────────────────────

class FinBERTAnalyzer:
    MODEL_NAME = "ProsusAI/finbert"
    LABEL_MAP  = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self):
        self._pipe = None
        self._loaded = False
        self._error: Optional[str] = None

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import pipeline
            self._pipe = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                top_k=None,               # return all class scores
                truncation=True,
                max_length=512,
            )
            self._loaded = True
            logger.info("FinBERT loaded successfully")
        except Exception as exc:
            self._error = str(exc)
            logger.warning("FinBERT load failed: %s", exc)
            self._loaded = True   # mark as attempted so we don't retry

    @property
    def available(self) -> bool:
        self._load()
        return self._pipe is not None

    def analyze(self, text: str) -> dict:
        self._load()
        if self._pipe is None:
            return {}
        text = _clean(text)
        if not text:
            return {}
        try:
            results = self._pipe(text)[0]  # list of {label, score}
            scores = {r["label"].lower(): r["score"] for r in results}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)
            label = max(scores, key=scores.get)
            compound = pos - neg  # [-1, 1]
            return {
                "label":    label,
                "positive": round(pos, 4),
                "negative": round(neg, 4),
                "neutral":  round(neu, 4),
                "compound": round(compound, 4),
                "model":    "finbert",
            }
        except Exception as exc:
            logger.warning("FinBERT inference error: %s", exc)
            return {}

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        self._load()
        if self._pipe is None:
            return [{}] * len(texts)
        cleaned = [_clean(t) for t in texts]
        results = []
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            try:
                batch_out = self._pipe(batch)
                for item_results in batch_out:
                    scores = {r["label"].lower(): r["score"] for r in item_results}
                    pos = scores.get("positive", 0.0)
                    neg = scores.get("negative", 0.0)
                    neu = scores.get("neutral",  0.0)
                    label = max(scores, key=scores.get)
                    results.append({
                        "label":    label,
                        "positive": round(pos, 4),
                        "negative": round(neg, 4),
                        "neutral":  round(neu, 4),
                        "compound": round(pos - neg, 4),
                        "model":    "finbert",
                    })
            except Exception as exc:
                logger.warning("FinBERT batch error: %s", exc)
                results.extend([{}] * len(batch))
        return results


# ── VADER wrapper ─────────────────────────────────────────────────────────────

class VADERAnalyzer:
    def __init__(self):
        self._sid = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._sid = SentimentIntensityAnalyzer()
        except Exception as exc:
            logger.warning("VADER load failed: %s", exc)
        self._loaded = True

    @property
    def available(self) -> bool:
        self._load()
        return self._sid is not None

    def analyze(self, text: str) -> dict:
        self._load()
        if self._sid is None:
            return {}
        text = _clean(text)
        if not text:
            return {}
        sc = self._sid.polarity_scores(text)
        compound = sc["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        pos = sc["pos"]
        neg = sc["neg"]
        neu = sc["neu"]
        return {
            "label":    label,
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral":  round(neu, 4),
            "compound": round(compound, 4),
            "model":    "vader",
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]


# ── Ensemble analyzer ─────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Primary: FinBERT (financial-domain BERT).
    Fallback: VADER (rule-based) when FinBERT unavailable.
    Both models always run so the UI can compare results.
    """

    def __init__(self):
        self.finbert = FinBERTAnalyzer()
        self.vader   = VADERAnalyzer()

    def analyze(self, text: str) -> dict:
        fb = self.finbert.analyze(text)
        vd = self.vader.analyze(text)
        primary = fb if fb else vd
        return {
            "finbert": fb,
            "vader":   vd,
            **primary,
            "model": "finbert" if fb else "vader",
        }

    def analyze_batch_finbert(
        self, texts: list[str], batch_size: int = 16
    ) -> list[dict]:
        return self.finbert.analyze_batch(texts, batch_size)

    def analyze_batch_vader(self, texts: list[str]) -> list[dict]:
        return self.vader.analyze_batch(texts)

    def finbert_available(self) -> bool:
        return self.finbert.available

    def vader_available(self) -> bool:
        return self.vader.available


# ── Model Evaluator (Financial PhraseBank) ────────────────────────────────────

class ModelEvaluator:
    """Evaluate FinBERT and VADER on the Financial PhraseBank benchmark."""

    PHRASEBANK_LABEL_MAP = {
        "positive": "positive",
        "negative": "negative",
        "neutral":  "neutral",
    }

    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer

    def load_phrasebank(self, split: str = "train", n: int = 500) -> tuple[list[str], list[str]]:
        """Load Financial PhraseBank from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "takala/financial_phrasebank",
                "sentences_allagree",
                trust_remote_code=True,
            )
            data = ds[split]
            texts  = [row["sentence"] for row in data][:n]
            labels = [
                ["negative", "neutral", "positive"][row["label"]]
                for row in data
            ][:n]
            return texts, labels
        except Exception:
            # Fallback: embedded sample
            return self._fallback_sample()

    def _fallback_sample(self) -> tuple[list[str], list[str]]:
        samples = [
            ("The company reported record profits of $2.3 billion this quarter, beating analyst expectations.", "positive"),
            ("Revenues surged 45% year-over-year, driven by strong demand in emerging markets.", "positive"),
            ("The board approved a 20% increase in dividend payments to shareholders.", "positive"),
            ("Our new product launch exceeded targets with 1 million units sold in the first month.", "positive"),
            ("The firm secured a landmark contract worth $500 million with the government.", "positive"),
            ("Operating margins improved to 18%, the highest level in five years.", "positive"),
            ("The company announced a buyback of $1 billion worth of shares.", "positive"),
            ("Earnings per share rose 30 cents to $1.85, well above the consensus estimate.", "positive"),
            ("The acquisition is expected to be immediately accretive to earnings.", "positive"),
            ("Strong cash generation allowed the company to reduce debt by $400 million.", "positive"),
            ("The company reported a net loss of $340 million, wider than expected.", "negative"),
            ("Revenue fell 22% as customer spending dried up amid the economic slowdown.", "negative"),
            ("The firm issued a profit warning, citing weaker-than-expected demand.", "negative"),
            ("Shares fell 12% after the company reported disappointing quarterly results.", "negative"),
            ("The company will cut 8% of its global workforce, affecting 4,200 employees.", "negative"),
            ("Write-downs of $1.2 billion dragged the quarterly results into the red.", "negative"),
            ("The credit rating was downgraded to junk status by Moody's.", "negative"),
            ("Regulatory investigations into the company's accounting practices have intensified.", "negative"),
            ("The company defaulted on its bond payments, triggering cross-default clauses.", "negative"),
            ("Free cash flow turned negative as capital expenditure rose sharply.", "negative"),
            ("The company maintained its annual guidance range of $4.50 to $4.75 per share.", "neutral"),
            ("The board met on Monday to discuss the quarterly financial results.", "neutral"),
            ("The company will release its first-quarter results on April 28th.", "neutral"),
            ("The firm operates in 42 countries with approximately 85,000 employees.", "neutral"),
            ("The new CEO will assume the role effective January 1st.", "neutral"),
            ("Trading volumes were in line with seasonal expectations.", "neutral"),
            ("The company reiterated its previously stated capital allocation policy.", "neutral"),
            ("The annual general meeting is scheduled for June 15th in Helsinki.", "neutral"),
            ("The agreement is subject to regulatory approval in multiple jurisdictions.", "neutral"),
            ("The company has 12% market share in the domestic segment.", "neutral"),
        ]
        texts  = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        return texts, labels

    def evaluate(self, n_samples: int = 500) -> dict:
        from sklearn.metrics import (
            accuracy_score, classification_report,
            confusion_matrix, roc_auc_score,
            matthews_corrcoef, cohen_kappa_score,
            average_precision_score,
        )
        import numpy as np

        texts, true_labels = self.load_phrasebank(n=n_samples)
        label_order = ["positive", "negative", "neutral"]

        # FinBERT
        fb_preds = []
        fb_probs = []
        fb_results = self.analyzer.analyze_batch_finbert(texts)
        for r in fb_results:
            if r:
                fb_preds.append(r.get("label", "neutral"))
                fb_probs.append([r.get("positive", 0), r.get("negative", 0), r.get("neutral", 0)])
            else:
                fb_preds.append("neutral")
                fb_probs.append([0.0, 0.0, 1.0])

        # VADER
        vd_preds  = []
        vd_results = self.analyzer.analyze_batch_vader(texts)
        for r in vd_results:
            vd_preds.append(r.get("label", "neutral") if r else "neutral")

        def compute_metrics(preds, probs=None, name=""):
            from sklearn.preprocessing import label_binarize
            report = classification_report(
                true_labels, preds, labels=label_order, output_dict=True
            )
            cm = confusion_matrix(true_labels, preds, labels=label_order)
            mcc = matthews_corrcoef(true_labels, preds)
            kappa = cohen_kappa_score(true_labels, preds)

            roc_auc = None
            avg_prec = None
            if probs is not None:
                probs_arr = np.array(probs)
                y_bin = label_binarize(true_labels, classes=label_order)
                try:
                    roc_auc = roc_auc_score(y_bin, probs_arr, multi_class="ovr", average="macro")
                    avg_prec = average_precision_score(y_bin, probs_arr, average="macro")
                except Exception:
                    pass

            return {
                "model":          name,
                "accuracy":       report["accuracy"],
                "macro_f1":       report["macro avg"]["f1-score"],
                "macro_precision":report["macro avg"]["precision"],
                "macro_recall":   report["macro avg"]["recall"],
                "weighted_f1":    report["weighted avg"]["f1-score"],
                "mcc":            mcc,
                "kappa":          kappa,
                "roc_auc_macro":  roc_auc,
                "avg_precision":  avg_prec,
                "per_class":      {
                    cls: {
                        "precision": report[cls]["precision"],
                        "recall":    report[cls]["recall"],
                        "f1":        report[cls]["f1-score"],
                        "support":   report[cls]["support"],
                    }
                    for cls in label_order if cls in report
                },
                "confusion_matrix": cm.tolist(),
                "labels":           label_order,
                "n_samples":        len(texts),
                "probs":            probs,
                "true_labels":      true_labels,
                "pred_labels":      preds,
            }

        fb_metrics = compute_metrics(fb_preds, fb_probs, "FinBERT")
        vd_metrics = compute_metrics(vd_preds,  None,    "VADER")

        return {"finbert": fb_metrics, "vader": vd_metrics}
