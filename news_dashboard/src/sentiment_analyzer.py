"""
Sentiment analysis via HuggingFace Inference API (FinBERT) + VADER fallback.

No local torch/transformers download required — FinBERT runs in the cloud.
Model: ProsusAI/finbert  (positive / negative / neutral)
"""

from __future__ import annotations
import re
import time
import logging
from typing import Optional

import requests
import numpy as np

logger = logging.getLogger(__name__)

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")

def _clean(text: str) -> str:
    text = _HTML_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", text).strip()[:512]


# ── HuggingFace Inference API ────────────────────────────────────────────────

class FinBERTAnalyzer:
    """
    Calls the HuggingFace serverless Inference API for ProsusAI/finbert.
    No local model download needed.  Works without an HF token (lower rate
    limit); set HF_TOKEN in Streamlit secrets for higher throughput.
    """

    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

    def __init__(self, hf_token: str = ""):
        self._token   = hf_token
        self._headers = ({"Authorization": f"Bearer {hf_token}"} if hf_token else {})
        self._status: Optional[str] = None   # "ok" | "loading" | "error"

    # ── low-level ────────────────────────────────────────────────────────────

    def _post(self, payload: dict, timeout: int = 60) -> Optional[dict | list]:
        try:
            resp = requests.post(
                self.HF_URL, headers=self._headers,
                json=payload, timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning("HF API request error: %s", exc)
            return None

        if resp.status_code == 200:
            self._status = "ok"
            return resp.json()

        if resp.status_code == 503:
            body = resp.json()
            wait = min(float(body.get("estimated_time", 20)), 30)
            logger.info("FinBERT model loading, waiting %.0fs…", wait)
            self._status = "loading"
            time.sleep(wait)
            return None   # caller will retry

        logger.warning("HF API %d: %s", resp.status_code, resp.text[:120])
        self._status = "error"
        return None

    @staticmethod
    def _parse_single(raw: list) -> dict:
        """Turn [[{label, score}, …]] or [{label, score}, …] into our dict."""
        items = raw[0] if (raw and isinstance(raw[0], list)) else raw
        scores = {r["label"].lower(): r["score"] for r in items}
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral",  0.0)
        label = max(scores, key=scores.get)
        return {
            "label":    label,
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral":  round(neu, 4),
            "compound": round(pos - neg, 4),
            "model":    "finbert",
        }

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        if self._status == "ok":
            return True
        r = self._post({"inputs": "market", "options": {"wait_for_model": True}})
        return r is not None

    def analyze(self, text: str) -> dict:
        text = _clean(text)
        if not text:
            return {}
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        raw = self._post(payload, timeout=60)
        if raw is None:
            # one retry after model warms up
            raw = self._post(payload, timeout=60)
        if raw is None:
            return {}
        try:
            return self._parse_single(raw)
        except Exception as exc:
            logger.warning("HF API parse error: %s  raw=%s", exc, str(raw)[:100])
            return {}

    def analyze_batch(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        cleaned = [_clean(t) for t in texts]
        results: list[dict] = []
        for i in range(0, len(cleaned), batch_size):
            batch = [t for t in cleaned[i: i + batch_size] if t]
            if not batch:
                results.extend([{}] * min(batch_size, len(cleaned) - i))
                continue
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            raw = self._post(payload, timeout=90)
            if raw is None:
                raw = self._post(payload, timeout=90)
            if raw is None:
                results.extend([{}] * len(batch))
                continue
            # batch response: list of (list of {label,score})
            try:
                for item in raw:
                    results.append(self._parse_single(item if isinstance(item[0], dict) else [item]))
            except Exception as exc:
                logger.warning("HF batch parse error: %s", exc)
                results.extend([{}] * len(batch))
        return results


# ── VADER ────────────────────────────────────────────────────────────────────

class VADERAnalyzer:
    def __init__(self):
        self._sid    = None
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
        if not self._sid:
            return {}
        text = _clean(text)
        if not text:
            return {}
        sc = self._sid.polarity_scores(text)
        c  = sc["compound"]
        label = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        return {
            "label":    label,
            "positive": round(sc["pos"], 4),
            "negative": round(sc["neg"], 4),
            "neutral":  round(sc["neu"], 4),
            "compound": round(c, 4),
            "model":    "vader",
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]


# ── Unified analyzer ──────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Primary:  FinBERT via HuggingFace Inference API (cloud, no GPU needed).
    Fallback: VADER (local, rule-based, always available).
    """

    def __init__(self, hf_token: str = ""):
        self.finbert = FinBERTAnalyzer(hf_token=hf_token)
        self.vader   = VADERAnalyzer()

    def analyze(self, text: str) -> dict:
        fb = self.finbert.analyze(text)
        vd = self.vader.analyze(text)
        primary = fb if fb else vd
        return {**primary, "finbert": fb, "vader": vd}

    def analyze_batch_finbert(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        return self.finbert.analyze_batch(texts, batch_size=batch_size)

    def analyze_batch_vader(self, texts: list[str]) -> list[dict]:
        return self.vader.analyze_batch(texts)

    def finbert_available(self) -> bool:
        return self.finbert.available

    def vader_available(self) -> bool:
        return self.vader.available


# ── Evaluator ────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Evaluate FinBERT and VADER on Financial PhraseBank (embedded subset)."""

    LABELS = ["positive", "negative", "neutral"]

    PHRASEBANK = [
        # positive
        ("The company reported record profits of $2.3 billion, beating analyst expectations.", "positive"),
        ("Revenues surged 45% year-over-year, driven by strong demand in emerging markets.", "positive"),
        ("The board approved a 20% increase in dividend payments to shareholders.", "positive"),
        ("Operating margins improved to 18%, the highest level in five years.", "positive"),
        ("The firm secured a landmark contract worth $500 million.", "positive"),
        ("Earnings per share rose 30 cents to $1.85, well above consensus.", "positive"),
        ("The company announced a buyback of $1 billion worth of shares.", "positive"),
        ("Strong cash generation allowed the company to reduce debt by $400 million.", "positive"),
        ("The acquisition is expected to be immediately accretive to earnings.", "positive"),
        ("Net income doubled to $890 million on the back of surging product demand.", "positive"),
        ("The firm raised full-year guidance citing robust demand and cost savings.", "positive"),
        ("New product launch exceeded targets with 1 million units sold in the first month.", "positive"),
        ("The stock hit a 52-week high after stellar quarterly results.", "positive"),
        ("Free cash flow improved markedly, enabling accelerated debt repayment.", "positive"),
        ("Customer acquisition grew 32% as the loyalty programme gained traction.", "positive"),
        # negative
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
        ("Management warned that supply-chain disruptions would persist through the year.", "negative"),
        ("The CEO resigned unexpectedly amid a boardroom dispute over strategy.", "negative"),
        ("Plant shutdown will reduce output by 30% and cost $150 million in losses.", "negative"),
        ("Gross margin compression continued as raw material costs soared.", "negative"),
        ("The firm faces potential fines of up to $2 billion following regulatory review.", "negative"),
        # neutral
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
        ("The firm announced it will report earnings before the market opens.", "neutral"),
        ("Production guidance for the full year remains unchanged at 2.4 million units.", "neutral"),
        ("The transaction is expected to close in the second quarter of fiscal 2025.", "neutral"),
        ("Management confirmed it is reviewing strategic options for the division.", "neutral"),
        ("The company hired a new chief financial officer effective next month.", "neutral"),
    ]

    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer

    def evaluate(self, n_samples: int = 45) -> dict:
        from sklearn.metrics import (
            accuracy_score, classification_report,
            confusion_matrix, roc_auc_score,
            matthews_corrcoef, cohen_kappa_score,
            average_precision_score,
        )
        from sklearn.preprocessing import label_binarize

        data   = self.PHRASEBANK[:n_samples]
        texts  = [d[0] for d in data]
        y_true = [d[1] for d in data]

        # FinBERT
        fb_results = self.analyzer.analyze_batch_finbert(texts)
        fb_preds, fb_probs = [], []
        for r in fb_results:
            if r:
                fb_preds.append(r["label"])
                fb_probs.append([r["positive"], r["negative"], r["neutral"]])
            else:
                fb_preds.append("neutral")
                fb_probs.append([0.0, 0.0, 1.0])

        # VADER
        vd_results = self.analyzer.analyze_batch_vader(texts)
        vd_preds   = [r["label"] if r else "neutral" for r in vd_results]

        def _metrics(preds, probs_arr, name):
            report = classification_report(
                y_true, preds, labels=self.LABELS, output_dict=True, zero_division=0
            )
            cm    = confusion_matrix(y_true, preds, labels=self.LABELS)
            mcc   = matthews_corrcoef(y_true, preds)
            kappa = cohen_kappa_score(y_true, preds)

            roc_auc = avg_prec = None
            if probs_arr is not None:
                y_bin = label_binarize(y_true, classes=self.LABELS)
                try:
                    roc_auc  = roc_auc_score(y_bin, probs_arr, multi_class="ovr", average="macro")
                    avg_prec = average_precision_score(y_bin, probs_arr, average="macro")
                except Exception:
                    pass

            return {
                "model":            name,
                "accuracy":         report["accuracy"],
                "macro_f1":         report["macro avg"]["f1-score"],
                "macro_precision":  report["macro avg"]["precision"],
                "macro_recall":     report["macro avg"]["recall"],
                "weighted_f1":      report["weighted avg"]["f1-score"],
                "mcc":              mcc,
                "kappa":            kappa,
                "roc_auc_macro":    roc_auc,
                "avg_precision":    avg_prec,
                "per_class": {
                    cls: {
                        "precision": report[cls]["precision"],
                        "recall":    report[cls]["recall"],
                        "f1":        report[cls]["f1-score"],
                        "support":   int(report[cls]["support"]),
                    }
                    for cls in self.LABELS if cls in report
                },
                "confusion_matrix": cm.tolist(),
                "labels":           self.LABELS,
                "n_samples":        len(texts),
                "probs":            probs_arr.tolist() if probs_arr is not None else None,
                "true_labels":      y_true,
                "pred_labels":      preds,
            }

        fb_probs_arr = np.array(fb_probs)
        return {
            "finbert": _metrics(fb_preds, fb_probs_arr, "FinBERT"),
            "vader":   _metrics(vd_preds, None,         "VADER"),
        }
