"""Feature engineering pipeline for fraud detection.

Creates features from raw transaction data including:
- Velocity features (transaction frequency, amount velocity)
- Aggregation features (rolling statistics by entity)
- Time-based features (hour, day, weekend patterns)
- Behavioral features (deviation from normal patterns)
- Risk indicators (country mismatches, suspicious domains)
"""

from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    
    Creates rich feature sets from raw transaction data by computing:
    - Entity-level aggregations (card, merchant, customer)
    - Time-windowed statistics (1h, 24h, 7d)
    - Behavioral anomaly indicators
    - Temporal patterns
    """
    
    # Time windows for aggregations (in hours)
    TIME_WINDOWS = {
        "1h": 1,
        "6h": 6,
        "24h": 24,
        "7d": 24 * 7,
        "30d": 24 * 30,
    }
    
    # Suspicious email domains
    SUSPICIOUS_DOMAINS = {
        "tempmail.com", "guerrillamail.com", "10minutemail.com", 
        "throwaway.email", "mailinator.com", "fakeinbox.com",
        "sharklasers.com", "yopmail.com"
    }
    
    # High-risk merchant categories
    HIGH_RISK_CATEGORIES = {
        "jewelry", "luxury", "electronics", "travel", "gambling",
        "cryptocurrency", "wire_transfer", "gift_cards"
    }
    
    def __init__(
        self,
        compute_velocity: bool = True,
        compute_aggregations: bool = True,
        compute_temporal: bool = True,
        compute_risk_indicators: bool = True,
    ) -> None:
        """
        Initialize feature engineer.
        
        Args:
            compute_velocity: Compute velocity features.
            compute_aggregations: Compute aggregation features.
            compute_temporal: Compute temporal features.
            compute_risk_indicators: Compute risk indicator features.
        """
        self.compute_velocity = compute_velocity
        self.compute_aggregations = compute_aggregations
        self.compute_temporal = compute_temporal
        self.compute_risk_indicators = compute_risk_indicators
        
        self._feature_names: list[str] = []
    
    @property
    def feature_names(self) -> list[str]:
        """Get list of engineered feature names."""
        return self._feature_names.copy()
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for dataset.
        
        Args:
            df: Raw transaction DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        logger.info("Starting feature engineering", rows=len(df))
        
        # Ensure timestamp is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp for windowed calculations
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        features = pd.DataFrame(index=df.index)
        
        # Base features
        features["amount"] = df["amount"]
        features["amount_log"] = np.log1p(df["amount"])
        
        # Temporal features
        if self.compute_temporal:
            features = pd.concat([features, self._compute_temporal_features(df)], axis=1)
        
        # Risk indicators
        if self.compute_risk_indicators:
            features = pd.concat([features, self._compute_risk_indicators(df)], axis=1)
        
        # Velocity features (requires sorted data)
        if self.compute_velocity:
            features = pd.concat([features, self._compute_velocity_features(df)], axis=1)
        
        # Aggregation features
        if self.compute_aggregations:
            features = pd.concat([features, self._compute_aggregation_features(df)], axis=1)
        
        # Store feature names
        self._feature_names = features.columns.tolist()
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        logger.info(
            "Feature engineering complete",
            n_features=len(self._feature_names),
            features=self._feature_names[:10],  # Log first 10
        )
        
        return features
    
    def _compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        ts = df["timestamp"]
        
        features = pd.DataFrame(index=df.index)
        
        # Hour of day (cyclical encoding)
        hour = ts.dt.hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (cyclical encoding)
        dow = ts.dt.dayofweek
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        # Binary indicators
        features["is_weekend"] = (dow >= 5).astype(int)
        features["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
        features["is_business_hours"] = ((hour >= 9) & (hour <= 17) & (dow < 5)).astype(int)
        
        # Day of month (for payday patterns)
        features["day_of_month"] = ts.dt.day
        features["is_month_start"] = (ts.dt.day <= 3).astype(int)
        features["is_month_end"] = (ts.dt.day >= 28).astype(int)
        
        return features
    
    def _compute_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk indicator features."""
        features = pd.DataFrame(index=df.index)
        
        # Email domain risk
        features["suspicious_email"] = df["email_domain"].isin(self.SUSPICIOUS_DOMAINS).astype(int)
        
        # Country mismatch indicators
        if "billing_country" in df.columns and "shipping_country" in df.columns:
            features["country_mismatch"] = (
                df["billing_country"] != df["shipping_country"]
            ).astype(int)
        
        if "card_country" in df.columns and "billing_country" in df.columns:
            features["card_billing_mismatch"] = (
                df["card_country"] != df["billing_country"]
            ).astype(int)
        
        # High-risk category
        if "merchant_category" in df.columns:
            features["high_risk_category"] = df["merchant_category"].isin(
                self.HIGH_RISK_CATEGORIES
            ).astype(int)
        
        # Online transaction (higher risk)
        features["is_online"] = df["is_online"].astype(int)
        
        # Amount-based risk
        features["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)
        features["is_large_amount"] = (df["amount"] > 1000).astype(int)
        features["is_very_large_amount"] = (df["amount"] > 5000).astype(int)
        
        return features
    
    def _compute_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocity and frequency features."""
        features = pd.DataFrame(index=df.index)
        
        # For each entity type, compute velocity metrics
        entity_cols = ["card_id", "customer_id", "merchant_id"]
        
        for entity in entity_cols:
            if entity not in df.columns:
                continue
            
            entity_name = entity.replace("_id", "")
            
            # Time since last transaction for this entity
            df_sorted = df.sort_values(["timestamp"])
            entity_groups = df_sorted.groupby(entity)["timestamp"]
            
            time_diff = entity_groups.diff().dt.total_seconds() / 3600  # hours
            features[f"{entity_name}_hours_since_last"] = time_diff.fillna(9999)
            
            # Transaction count in rolling windows
            for window_name, hours in self.TIME_WINDOWS.items():
                if hours <= 24 * 7:  # Only compute for windows <= 7 days
                    features[f"{entity_name}_txn_count_{window_name}"] = self._rolling_count(
                        df, entity, hours
                    )
        
        return features
    
    def _compute_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregation features per entity."""
        features = pd.DataFrame(index=df.index)
        
        entity_cols = ["card_id", "merchant_id"]
        
        for entity in entity_cols:
            if entity not in df.columns:
                continue
            
            entity_name = entity.replace("_id", "")
            
            # Historical statistics for entity
            for window_name, hours in [("24h", 24), ("7d", 24 * 7)]:
                # Amount statistics
                amount_stats = self._rolling_amount_stats(df, entity, hours)
                
                features[f"{entity_name}_amount_mean_{window_name}"] = amount_stats["mean"]
                features[f"{entity_name}_amount_std_{window_name}"] = amount_stats["std"]
                features[f"{entity_name}_amount_max_{window_name}"] = amount_stats["max"]
                
                # Z-score of current amount vs rolling mean
                features[f"{entity_name}_amount_zscore_{window_name}"] = (
                    (df["amount"] - amount_stats["mean"]) / 
                    (amount_stats["std"] + 1e-6)
                )
        
        # Merchant diversity for card
        if "card_id" in df.columns and "merchant_id" in df.columns:
            for window_name, hours in [("24h", 24), ("7d", 24 * 7)]:
                features[f"card_unique_merchants_{window_name}"] = self._rolling_unique_count(
                    df, "card_id", "merchant_id", hours
                )
        
        return features
    
    def _rolling_count(
        self, 
        df: pd.DataFrame, 
        entity_col: str, 
        window_hours: int
    ) -> pd.Series:
        """Count transactions per entity in rolling window."""
        result = pd.Series(0, index=df.index)
        
        # Group by entity
        for entity_val, group in df.groupby(entity_col):
            group = group.sort_values("timestamp")
            
            counts = []
            for idx, row in group.iterrows():
                window_start = row["timestamp"] - pd.Timedelta(hours=window_hours)
                count = ((group["timestamp"] >= window_start) & 
                        (group["timestamp"] < row["timestamp"])).sum()
                counts.append((idx, count))
            
            for idx, count in counts:
                result.loc[idx] = count
        
        return result
    
    def _rolling_amount_stats(
        self, 
        df: pd.DataFrame, 
        entity_col: str, 
        window_hours: int
    ) -> dict[str, pd.Series]:
        """Compute rolling amount statistics per entity."""
        mean_result = pd.Series(0.0, index=df.index)
        std_result = pd.Series(0.0, index=df.index)
        max_result = pd.Series(0.0, index=df.index)
        
        for entity_val, group in df.groupby(entity_col):
            group = group.sort_values("timestamp")
            
            for idx, row in group.iterrows():
                window_start = row["timestamp"] - pd.Timedelta(hours=window_hours)
                mask = (group["timestamp"] >= window_start) & (group["timestamp"] < row["timestamp"])
                window_amounts = group.loc[mask, "amount"]
                
                if len(window_amounts) > 0:
                    mean_result.loc[idx] = window_amounts.mean()
                    std_result.loc[idx] = window_amounts.std() if len(window_amounts) > 1 else 0
                    max_result.loc[idx] = window_amounts.max()
        
        return {"mean": mean_result, "std": std_result, "max": max_result}
    
    def _rolling_unique_count(
        self, 
        df: pd.DataFrame, 
        entity_col: str, 
        target_col: str,
        window_hours: int
    ) -> pd.Series:
        """Count unique values of target column per entity in rolling window."""
        result = pd.Series(0, index=df.index)
        
        for entity_val, group in df.groupby(entity_col):
            group = group.sort_values("timestamp")
            
            for idx, row in group.iterrows():
                window_start = row["timestamp"] - pd.Timedelta(hours=window_hours)
                mask = (group["timestamp"] >= window_start) & (group["timestamp"] < row["timestamp"])
                unique_count = group.loc[mask, target_col].nunique()
                result.loc[idx] = unique_count
        
        return result
    
    def get_feature_importance_names(self) -> dict[str, str]:
        """Get human-readable names for features (for explainability)."""
        return {
            "amount": "Transaction Amount",
            "amount_log": "Log Transaction Amount",
            "hour_sin": "Hour (Sine)",
            "hour_cos": "Hour (Cosine)",
            "is_weekend": "Weekend Transaction",
            "is_night": "Night Transaction (10pm-5am)",
            "suspicious_email": "Suspicious Email Domain",
            "country_mismatch": "Billing/Shipping Country Mismatch",
            "card_billing_mismatch": "Card/Billing Country Mismatch",
            "high_risk_category": "High-Risk Merchant Category",
            "is_round_amount": "Round Amount ($X00)",
            "is_large_amount": "Large Amount (>$1000)",
            "card_hours_since_last": "Hours Since Last Card Transaction",
            "card_txn_count_1h": "Card Transactions in Last Hour",
            "card_txn_count_24h": "Card Transactions in Last 24h",
            "card_amount_zscore_24h": "Amount Z-Score vs 24h Average",
            "card_unique_merchants_24h": "Unique Merchants in 24h",
        }


class FastFeatureEngineer:
    """
    Memory-efficient feature engineer for large datasets.
    
    Uses vectorized operations and avoids expensive groupby operations
    for production batch processing.
    """
    
    def __init__(self) -> None:
        """Initialize fast feature engineer."""
        self._feature_names: list[str] = []
    
    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names.copy()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast feature transformation for scoring.
        
        Only computes features that don't require historical lookups.
        For full features, use FeatureEngineer with pre-computed aggregations.
        """
        features = pd.DataFrame(index=df.index)
        
        # Amount features
        features["amount"] = df["amount"]
        features["amount_log"] = np.log1p(df["amount"])
        
        # Temporal (vectorized)
        ts = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour
        dow = ts.dt.dayofweek
        
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        features["is_weekend"] = (dow >= 5).astype(int)
        features["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
        
        # Risk indicators (vectorized)
        suspicious_domains = {
            "tempmail.com", "guerrillamail.com", "10minutemail.com", 
            "throwaway.email", "mailinator.com"
        }
        features["suspicious_email"] = df["email_domain"].isin(suspicious_domains).astype(int)
        features["is_online"] = df["is_online"].astype(int)
        features["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)
        features["is_large_amount"] = (df["amount"] > 1000).astype(int)
        
        if "billing_country" in df.columns and "shipping_country" in df.columns:
            features["country_mismatch"] = (
                df["billing_country"] != df["shipping_country"]
            ).astype(int)
        
        self._feature_names = features.columns.tolist()
        
        return features.fillna(0)
