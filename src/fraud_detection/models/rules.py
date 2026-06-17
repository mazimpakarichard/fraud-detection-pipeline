"""Rule-based fraud detection model.

Implements deterministic rules based on domain knowledge:
- Amount thresholds
- Velocity limits
- Geographic patterns
- Time-based patterns
- Risk indicators

Each rule returns a score 0-1 and reason code for explainability.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RuleResult:
    """Result from a single rule evaluation."""

    rule_name: str
    score: float  # 0-1, higher = more anomalous
    triggered: bool
    reason: str | None = None


class RuleBasedModel:
    """
    Rule-based fraud detection using domain knowledge.

    Rules are based on common fraud patterns:
    - High-value transactions
    - Unusual transaction timing
    - Velocity anomalies
    - Geographic inconsistencies
    - Suspicious characteristics
    """

    def __init__(
        self,
        amount_threshold: float = 5000.0,
        velocity_threshold_1h: int = 5,
        velocity_threshold_24h: int = 20,
        night_hours: tuple[int, int] = (1, 5),
    ) -> None:
        """
        Initialize rule-based model.

        Args:
            amount_threshold: Flag transactions above this amount.
            velocity_threshold_1h: Max transactions per card per hour.
            velocity_threshold_24h: Max transactions per card per 24h.
            night_hours: Tuple of (start, end) hours for night activity.
        """
        self.amount_threshold = amount_threshold
        self.velocity_threshold_1h = velocity_threshold_1h
        self.velocity_threshold_24h = velocity_threshold_24h
        self.night_hours = night_hours

        self.rule_weights = {
            "high_amount": 0.25,
            "velocity_1h": 0.20,
            "velocity_24h": 0.10,
            "night_activity": 0.10,
            "country_mismatch": 0.15,
            "suspicious_email": 0.10,
            "high_risk_merchant": 0.10,
        }

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        """
        Score transactions using rule-based system.

        Args:
            df: DataFrame with transaction features.

        Returns:
            Tuple of (scores array, list of reason codes per transaction).
        """
        n_samples = len(df)
        scores = np.zeros(n_samples)
        all_reasons: list[list[dict[str, Any]]] = [[] for _ in range(n_samples)]

        # Rule 1: High amount
        if "amount" in df.columns:
            high_amount_mask = df["amount"] > self.amount_threshold
            rule_score = (df["amount"] / self.amount_threshold).clip(0, 1).values
            rule_score = np.where(high_amount_mask, rule_score, 0)
            scores += rule_score * self.rule_weights["high_amount"]

            for i in np.where(high_amount_mask)[0]:
                all_reasons[i].append(
                    {
                        "rule": "high_amount",
                        "score": float(rule_score[i]),
                        "reason": f"Amount ${df.iloc[i]['amount']:.2f} exceeds threshold ${self.amount_threshold:.2f}",
                    }
                )

        # Rule 2: Velocity - transactions in 1 hour
        if "card_txn_count_1h" in df.columns:
            velocity_1h = df["card_txn_count_1h"].values
            velocity_mask = velocity_1h >= self.velocity_threshold_1h
            rule_score = (velocity_1h / self.velocity_threshold_1h).clip(0, 1)
            rule_score = np.where(velocity_mask, rule_score, rule_score * 0.5)
            scores += rule_score * self.rule_weights["velocity_1h"]

            for i in np.where(velocity_mask)[0]:
                all_reasons[i].append(
                    {
                        "rule": "velocity_1h",
                        "score": float(rule_score[i]),
                        "reason": f"{int(velocity_1h[i])} transactions in last hour (threshold: {self.velocity_threshold_1h})",
                    }
                )

        # Rule 3: Velocity - transactions in 24 hours
        if "card_txn_count_24h" in df.columns:
            velocity_24h = df["card_txn_count_24h"].values
            velocity_mask = velocity_24h >= self.velocity_threshold_24h
            rule_score = (velocity_24h / self.velocity_threshold_24h).clip(0, 1)
            rule_score = np.where(velocity_mask, rule_score, rule_score * 0.3)
            scores += rule_score * self.rule_weights["velocity_24h"]

            for i in np.where(velocity_mask)[0]:
                all_reasons[i].append(
                    {
                        "rule": "velocity_24h",
                        "score": float(rule_score[i]),
                        "reason": f"{int(velocity_24h[i])} transactions in last 24h (threshold: {self.velocity_threshold_24h})",
                    }
                )

        # Rule 4: Night activity
        if "is_night" in df.columns:
            night_mask = df["is_night"].astype(bool).values
            # Night activity is suspicious only for high amounts
            if "amount" in df.columns:
                high_night_mask = night_mask & (df["amount"] > 500).values
                rule_score = np.where(high_night_mask, 0.8, np.where(night_mask, 0.3, 0))
            else:
                rule_score = np.where(night_mask, 0.5, 0)
            scores += rule_score * self.rule_weights["night_activity"]

            for i in np.where(night_mask & (rule_score > 0.3))[0]:
                all_reasons[i].append(
                    {
                        "rule": "night_activity",
                        "score": float(rule_score[i]),
                        "reason": "Transaction during unusual hours (1am-5am)",
                    }
                )

        # Rule 5: Country mismatch
        if "country_mismatch" in df.columns:
            mismatch_mask = df["country_mismatch"].astype(bool).values
            rule_score = np.where(mismatch_mask, 0.7, 0)
            scores += rule_score * self.rule_weights["country_mismatch"]

            for i in np.where(mismatch_mask)[0]:
                all_reasons[i].append(
                    {
                        "rule": "country_mismatch",
                        "score": 0.7,
                        "reason": "Billing and shipping countries do not match",
                    }
                )

        # Rule 6: Suspicious email
        if "suspicious_email" in df.columns:
            suspicious_mask = df["suspicious_email"].astype(bool).values
            rule_score = np.where(suspicious_mask, 0.8, 0)
            scores += rule_score * self.rule_weights["suspicious_email"]

            for i in np.where(suspicious_mask)[0]:
                all_reasons[i].append(
                    {
                        "rule": "suspicious_email",
                        "score": 0.8,
                        "reason": "Suspicious or disposable email domain detected",
                    }
                )

        # Rule 7: High-risk merchant category
        if "high_risk_category" in df.columns:
            high_risk_mask = df["high_risk_category"].astype(bool).values
            rule_score = np.where(high_risk_mask, 0.5, 0)
            scores += rule_score * self.rule_weights["high_risk_merchant"]

            for i in np.where(high_risk_mask)[0]:
                all_reasons[i].append(
                    {
                        "rule": "high_risk_merchant",
                        "score": 0.5,
                        "reason": "Transaction at high-risk merchant category",
                    }
                )

        # Normalize scores to 0-1 range
        scores = np.clip(scores, 0, 1)

        logger.info(
            "Rule-based scoring complete",
            n_samples=n_samples,
            mean_score=float(np.mean(scores)),
            flagged_count=int(np.sum(scores > 0.5)),
        )

        return scores, all_reasons

    def get_rule_descriptions(self) -> dict[str, str]:
        """Get human-readable descriptions of all rules."""
        return {
            "high_amount": f"Transaction amount exceeds ${self.amount_threshold:.0f}",
            "velocity_1h": f"More than {self.velocity_threshold_1h} transactions per hour from same card",
            "velocity_24h": f"More than {self.velocity_threshold_24h} transactions per 24h from same card",
            "night_activity": f"Transaction during unusual hours ({self.night_hours[0]}:00-{self.night_hours[1]}:00)",
            "country_mismatch": "Billing and shipping countries do not match",
            "suspicious_email": "Email from known disposable/suspicious domain",
            "high_risk_merchant": "Transaction at high-risk merchant category",
        }
