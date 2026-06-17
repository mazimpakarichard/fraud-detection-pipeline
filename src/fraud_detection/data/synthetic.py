"""Synthetic transaction data generator with injected anomalies.

Generates realistic transaction data with various types of fraud patterns:
- Amount outliers (unusually high transactions)
- Velocity anomalies (rapid succession transactions)
- Geographic anomalies (impossible travel)
- Off-hours activity bursts
- Duplicate/clone transactions
- Merchant category anomalies
"""

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly injection rates."""

    # Overall anomaly rate
    total_anomaly_rate: float = 0.02

    # Distribution of anomaly types (should sum to 1.0)
    amount_outlier_ratio: float = 0.25
    velocity_anomaly_ratio: float = 0.20
    geographic_anomaly_ratio: float = 0.15
    off_hours_ratio: float = 0.15
    duplicate_ratio: float = 0.10
    merchant_anomaly_ratio: float = 0.15

    def validate(self) -> None:
        """Validate configuration."""
        total = (
            self.amount_outlier_ratio
            + self.velocity_anomaly_ratio
            + self.geographic_anomaly_ratio
            + self.off_hours_ratio
            + self.duplicate_ratio
            + self.merchant_anomaly_ratio
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Anomaly ratios must sum to 1.0, got {total}")


@dataclass
class MerchantProfile:
    """Merchant characteristics for realistic transactions."""

    merchant_id: str
    category: str
    avg_amount: float
    std_amount: float
    min_amount: float
    max_amount: float
    online_ratio: float  # Probability of online transaction
    peak_hours: list[int] = field(default_factory=lambda: list(range(9, 21)))


class SyntheticTransactionGenerator:
    """
    Generate synthetic transaction data with configurable anomaly injection.

    Produces realistic transaction patterns based on:
    - Time-of-day distributions
    - Day-of-week patterns
    - Merchant category profiles
    - Geographic distributions
    - Card usage patterns
    """

    # Merchant category profiles with realistic parameters
    MERCHANT_PROFILES = [
        MerchantProfile("grocery_chain", "grocery", 65.0, 40.0, 5.0, 300.0, 0.15),
        MerchantProfile("gas_station", "gas_station", 45.0, 20.0, 10.0, 150.0, 0.05),
        MerchantProfile("restaurant", "food_beverage", 35.0, 25.0, 8.0, 200.0, 0.30),
        MerchantProfile("coffee_shop", "food_beverage", 8.0, 4.0, 3.0, 25.0, 0.40),
        MerchantProfile("electronics", "electronics", 250.0, 200.0, 20.0, 2000.0, 0.70),
        MerchantProfile("clothing", "apparel", 85.0, 60.0, 15.0, 500.0, 0.55),
        MerchantProfile("pharmacy", "health", 35.0, 30.0, 5.0, 200.0, 0.25),
        MerchantProfile("streaming", "digital_services", 15.0, 5.0, 5.0, 50.0, 1.0),
        MerchantProfile("utilities", "utilities", 120.0, 80.0, 30.0, 500.0, 0.90),
        MerchantProfile("travel", "travel", 350.0, 300.0, 50.0, 5000.0, 0.85),
        MerchantProfile("hotel", "lodging", 180.0, 120.0, 80.0, 1000.0, 0.75),
        MerchantProfile("rideshare", "transportation", 25.0, 15.0, 5.0, 100.0, 1.0),
        MerchantProfile("subscription", "digital_services", 12.0, 8.0, 5.0, 100.0, 1.0),
        MerchantProfile("jewelry", "luxury", 500.0, 800.0, 50.0, 10000.0, 0.40),
        MerchantProfile("furniture", "home", 400.0, 350.0, 50.0, 5000.0, 0.45),
    ]

    COUNTRIES = ["USA", "CAN", "GBR", "DEU", "FRA", "JPN", "AUS", "BRA", "MEX", "IND"]
    CARD_TYPES = ["VISA", "MASTERCARD", "AMEX", "DISCOVER"]
    EMAIL_DOMAINS = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "icloud.com",
        "aol.com",
        "protonmail.com",
        "mail.com",
    ]
    SUSPICIOUS_DOMAINS = [
        "tempmail.com",
        "guerrillamail.com",
        "10minutemail.com",
        "throwaway.email",
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        anomaly_config: AnomalyConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the generator.

        Args:
            settings: Application settings.
            anomaly_config: Anomaly injection configuration.
            seed: Random seed for reproducibility.
        """
        self.settings = settings or get_settings()
        self.anomaly_config = anomaly_config or AnomalyConfig()
        self.anomaly_config.validate()
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._merchant_pool: dict[str, list[str]] = {}
        self._card_pool: list[str] = []
        self._customer_pool: list[dict[str, Any]] = []

    def _generate_merchant_id(self, profile: MerchantProfile) -> str:
        """Generate a unique merchant ID for a profile."""
        if profile.category not in self._merchant_pool:
            self._merchant_pool[profile.category] = []

        # Create multiple merchants per category
        idx = len(self._merchant_pool[profile.category])
        merchant_id = f"{profile.merchant_id}_{idx:04d}"
        self._merchant_pool[profile.category].append(merchant_id)
        return merchant_id

    def _initialize_pools(
        self, n_cards: int, n_customers: int, n_merchants_per_category: int = 50
    ) -> None:
        """Initialize card, customer, and merchant pools."""
        # Generate merchants
        self._merchant_pool = {}
        for profile in self.MERCHANT_PROFILES:
            self._merchant_pool[profile.category] = [
                f"{profile.merchant_id}_{i:04d}" for i in range(n_merchants_per_category)
            ]

        # Generate cards
        self._card_pool = [f"card_{uuid.uuid4().hex[:12]}" for _ in range(n_cards)]

        # Generate customers with profiles
        self._customer_pool = []
        for i in range(n_customers):
            country = np.random.choice(
                self.COUNTRIES, p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]
            )
            self._customer_pool.append(
                {
                    "customer_id": f"cust_{i:08d}",
                    "home_country": country,
                    "email_domain": np.random.choice(self.EMAIL_DOMAINS),
                    "primary_card": self._card_pool[i % n_cards],
                    "typical_amount": np.random.lognormal(4.0, 1.0),  # ~$55 median
                }
            )

    def _generate_transaction_time(self, base_date: datetime, is_anomaly: bool = False) -> datetime:
        """Generate realistic transaction timestamp."""
        # Day of week distribution (more activity on weekdays)
        day_offset = np.random.choice(7, p=[0.12, 0.15, 0.15, 0.15, 0.18, 0.15, 0.10])

        if is_anomaly and np.random.random() < 0.3:
            # Off-hours anomaly: 1-5 AM
            hour = np.random.randint(1, 5)
        else:
            # Normal distribution centered around noon/evening
            hour_probs = np.zeros(24)
            for h in range(24):
                if 6 <= h <= 9:
                    hour_probs[h] = 0.06
                elif 10 <= h <= 14:
                    hour_probs[h] = 0.08
                elif 15 <= h <= 20:
                    hour_probs[h] = 0.10
                elif 21 <= h <= 23:
                    hour_probs[h] = 0.04
                else:
                    hour_probs[h] = 0.01
            hour_probs /= hour_probs.sum()
            hour = np.random.choice(24, p=hour_probs)

        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)

        return base_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

    def _generate_amount(self, profile: MerchantProfile, is_outlier: bool = False) -> float:
        """Generate transaction amount based on merchant profile."""
        if is_outlier:
            # Extreme outlier: 10-50x normal max
            return np.random.uniform(profile.max_amount * 10, profile.max_amount * 50)

        # Log-normal distribution for realistic amounts
        amount = np.random.lognormal(
            np.log(profile.avg_amount), profile.std_amount / profile.avg_amount
        )
        return np.clip(amount, profile.min_amount, profile.max_amount * 2)

    def _generate_ip_address(self, is_suspicious: bool = False) -> str:
        """Generate IP address."""
        if is_suspicious:
            # Known VPN/proxy ranges or Tor exit nodes
            suspicious_prefixes = ["185.220.", "104.244.", "45.33.", "103.21."]
            prefix = np.random.choice(suspicious_prefixes)
            return f"{prefix}{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"

        # Normal residential/commercial IPs
        octets = [
            np.random.randint(1, 224),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(1, 255),
        ]
        return ".".join(str(o) for o in octets)

    def _inject_velocity_anomaly(
        self, base_txn: dict[str, Any], count: int = 5
    ) -> list[dict[str, Any]]:
        """Generate rapid-fire transactions from same card."""
        transactions = []
        base_time = base_txn["timestamp"]

        for i in range(count):
            txn = base_txn.copy()
            txn["transaction_id"] = f"txn_{uuid.uuid4().hex}"
            # Transactions within minutes of each other
            txn["timestamp"] = base_time + timedelta(minutes=i * np.random.randint(1, 3))
            # Different merchants but similar category
            profile = np.random.choice(self.MERCHANT_PROFILES)
            txn["merchant_id"] = np.random.choice(
                self._merchant_pool.get(profile.category, ["unknown"])
            )
            txn["merchant_category"] = profile.category
            txn["amount"] = round(self._generate_amount(profile), 2)
            # Changing device/IP indicates card compromise
            txn["device_id"] = f"device_{uuid.uuid4().hex[:8]}"
            txn["ip_address"] = self._generate_ip_address(is_suspicious=True)
            txn["is_fraud"] = True
            txn["fraud_label_source"] = "INJECTED"
            transactions.append(txn)

        return transactions

    def _inject_geographic_anomaly(self, base_txn: dict[str, Any]) -> dict[str, Any]:
        """Generate impossible travel anomaly."""
        txn = base_txn.copy()
        # Billing and shipping in very different countries
        countries = list(set(self.COUNTRIES) - {base_txn.get("billing_country", "USA")})
        txn["shipping_country"] = np.random.choice(countries)
        txn["card_country"] = np.random.choice(countries)
        txn["ip_address"] = self._generate_ip_address(is_suspicious=True)
        txn["is_fraud"] = True
        txn["fraud_label_source"] = "INJECTED"
        return txn

    def _inject_duplicate(self, base_txn: dict[str, Any]) -> dict[str, Any]:
        """Generate near-duplicate transaction (double charge pattern)."""
        txn = base_txn.copy()
        txn["transaction_id"] = f"txn_{uuid.uuid4().hex}"
        # Same amount, same merchant, within seconds
        txn["timestamp"] = base_txn["timestamp"] + timedelta(seconds=np.random.randint(1, 30))
        txn["is_fraud"] = True
        txn["fraud_label_source"] = "INJECTED"
        return txn

    def _inject_merchant_anomaly(self, base_txn: dict[str, Any]) -> dict[str, Any]:
        """Generate unusual merchant category for card pattern."""
        txn = base_txn.copy()
        # High-risk categories unusual for the card
        high_risk_categories = ["jewelry", "luxury", "travel", "electronics"]
        profile = next(p for p in self.MERCHANT_PROFILES if p.category in high_risk_categories)
        txn["merchant_id"] = np.random.choice(
            self._merchant_pool.get(profile.category, ["unknown"])
        )
        txn["merchant_category"] = profile.category
        txn["amount"] = round(self._generate_amount(profile, is_outlier=True), 2)
        txn["email_domain"] = np.random.choice(self.SUSPICIOUS_DOMAINS)
        txn["is_fraud"] = True
        txn["fraud_label_source"] = "INJECTED"
        return txn

    def generate(
        self,
        n_transactions: int = 1_000_000,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        n_cards: int = 50_000,
        n_customers: int = 40_000,
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction dataset.

        Args:
            n_transactions: Number of transactions to generate.
            start_date: Start of transaction period.
            end_date: End of transaction period.
            n_cards: Number of unique cards.
            n_customers: Number of unique customers.

        Returns:
            DataFrame with synthetic transactions.
        """
        # Reset seed for reproducibility on each generate() call
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        logger.info(
            "Generating synthetic transactions",
            n_transactions=n_transactions,
            anomaly_rate=self.anomaly_config.total_anomaly_rate,
        )

        if start_date is None:
            # Use fixed reference date for reproducibility when seed is set
            ref_date = datetime(2024, 1, 1, 12, 0, 0) if self.seed is not None else datetime.now()
            start_date = ref_date - timedelta(days=90)
        if end_date is None:
            ref_date = datetime(2024, 1, 1, 12, 0, 0) if self.seed is not None else datetime.now()
            end_date = ref_date

        # Initialize entity pools
        self._initialize_pools(n_cards, n_customers)

        # Calculate anomaly counts
        n_anomalies = int(n_transactions * self.anomaly_config.total_anomaly_rate)
        n_normal = n_transactions - n_anomalies

        anomaly_counts = {
            "amount_outlier": int(n_anomalies * self.anomaly_config.amount_outlier_ratio),
            "velocity": int(n_anomalies * self.anomaly_config.velocity_anomaly_ratio),
            "geographic": int(n_anomalies * self.anomaly_config.geographic_anomaly_ratio),
            "off_hours": int(n_anomalies * self.anomaly_config.off_hours_ratio),
            "duplicate": int(n_anomalies * self.anomaly_config.duplicate_ratio),
            "merchant": int(n_anomalies * self.anomaly_config.merchant_anomaly_ratio),
        }

        transactions: list[dict[str, Any]] = []
        days_range = (end_date - start_date).days

        # Generate normal transactions
        logger.info("Generating normal transactions", count=n_normal)
        for _ in range(n_normal):
            profile = np.random.choice(self.MERCHANT_PROFILES)
            customer = np.random.choice(self._customer_pool)
            base_date = start_date + timedelta(days=np.random.randint(0, days_range))

            txn = {
                "transaction_id": f"txn_{uuid.uuid4().hex}",
                "timestamp": self._generate_transaction_time(base_date),
                "amount": round(self._generate_amount(profile), 2),
                "currency": "USD",
                "merchant_id": np.random.choice(
                    self._merchant_pool.get(profile.category, ["unknown"])
                ),
                "merchant_category": profile.category,
                "card_id": customer["primary_card"],
                "card_type": np.random.choice(self.CARD_TYPES, p=[0.45, 0.35, 0.12, 0.08]),
                "card_country": customer["home_country"],
                "customer_id": customer["customer_id"],
                "device_id": (
                    f"device_{hashlib.md5(customer['customer_id'].encode(), usedforsecurity=False).hexdigest()[:8]}"
                ),
                "ip_address": self._generate_ip_address(),
                "email_domain": customer["email_domain"],
                "billing_country": customer["home_country"],
                "shipping_country": customer["home_country"],
                "is_online": np.random.random() < profile.online_ratio,
                "product_category": profile.category,
                "data_source": "SYNTHETIC",
                "is_fraud": False,
                "fraud_label_source": "INJECTED",
            }
            transactions.append(txn)

        # Inject anomalies
        logger.info("Injecting anomalies", **anomaly_counts)

        # Amount outliers
        for _ in range(anomaly_counts["amount_outlier"]):
            profile = np.random.choice(self.MERCHANT_PROFILES)
            customer = np.random.choice(self._customer_pool)
            base_date = start_date + timedelta(days=np.random.randint(0, days_range))

            txn = {
                "transaction_id": f"txn_{uuid.uuid4().hex}",
                "timestamp": self._generate_transaction_time(base_date, is_anomaly=True),
                "amount": round(self._generate_amount(profile, is_outlier=True), 2),
                "currency": "USD",
                "merchant_id": np.random.choice(
                    self._merchant_pool.get(profile.category, ["unknown"])
                ),
                "merchant_category": profile.category,
                "card_id": customer["primary_card"],
                "card_type": np.random.choice(self.CARD_TYPES),
                "card_country": customer["home_country"],
                "customer_id": customer["customer_id"],
                "device_id": f"device_{uuid.uuid4().hex[:8]}",
                "ip_address": self._generate_ip_address(is_suspicious=True),
                "email_domain": np.random.choice(self.SUSPICIOUS_DOMAINS),
                "billing_country": customer["home_country"],
                "shipping_country": np.random.choice(self.COUNTRIES),
                "is_online": True,
                "product_category": profile.category,
                "data_source": "SYNTHETIC",
                "is_fraud": True,
                "fraud_label_source": "INJECTED",
            }
            transactions.append(txn)

        # Velocity anomalies (groups of rapid transactions)
        velocity_groups = anomaly_counts["velocity"] // 5
        for _ in range(velocity_groups):
            customer = np.random.choice(self._customer_pool)
            base_date = start_date + timedelta(days=np.random.randint(0, days_range))
            profile = np.random.choice(self.MERCHANT_PROFILES)

            base_txn = {
                "timestamp": self._generate_transaction_time(base_date, is_anomaly=True),
                "card_id": customer["primary_card"],
                "customer_id": customer["customer_id"],
                "card_type": np.random.choice(self.CARD_TYPES),
                "card_country": customer["home_country"],
                "billing_country": customer["home_country"],
                "shipping_country": customer["home_country"],
                "is_online": True,
                "currency": "USD",
                "data_source": "SYNTHETIC",
                "email_domain": customer["email_domain"],
            }
            transactions.extend(self._inject_velocity_anomaly(base_txn))

        # Geographic anomalies
        for _ in range(anomaly_counts["geographic"]):
            customer = np.random.choice(self._customer_pool)
            base_date = start_date + timedelta(days=np.random.randint(0, days_range))
            profile = np.random.choice(self.MERCHANT_PROFILES)

            base_txn = {
                "transaction_id": f"txn_{uuid.uuid4().hex}",
                "timestamp": self._generate_transaction_time(base_date, is_anomaly=True),
                "amount": round(self._generate_amount(profile), 2),
                "currency": "USD",
                "merchant_id": np.random.choice(
                    self._merchant_pool.get(profile.category, ["unknown"])
                ),
                "merchant_category": profile.category,
                "card_id": customer["primary_card"],
                "card_type": np.random.choice(self.CARD_TYPES),
                "card_country": customer["home_country"],
                "customer_id": customer["customer_id"],
                "device_id": f"device_{uuid.uuid4().hex[:8]}",
                "ip_address": self._generate_ip_address(),
                "email_domain": customer["email_domain"],
                "billing_country": customer["home_country"],
                "shipping_country": customer["home_country"],
                "is_online": True,
                "product_category": profile.category,
                "data_source": "SYNTHETIC",
                "is_fraud": False,
                "fraud_label_source": "INJECTED",
            }
            transactions.append(self._inject_geographic_anomaly(base_txn))

        # Off-hours anomalies
        for _ in range(anomaly_counts["off_hours"]):
            profile = np.random.choice(self.MERCHANT_PROFILES)
            customer = np.random.choice(self._customer_pool)
            base_date = start_date + timedelta(days=np.random.randint(0, days_range))

            txn = {
                "transaction_id": f"txn_{uuid.uuid4().hex}",
                "timestamp": self._generate_transaction_time(base_date, is_anomaly=True),
                "amount": round(self._generate_amount(profile) * np.random.uniform(2, 5), 2),
                "currency": "USD",
                "merchant_id": np.random.choice(
                    self._merchant_pool.get(profile.category, ["unknown"])
                ),
                "merchant_category": profile.category,
                "card_id": customer["primary_card"],
                "card_type": np.random.choice(self.CARD_TYPES),
                "card_country": customer["home_country"],
                "customer_id": customer["customer_id"],
                "device_id": f"device_{uuid.uuid4().hex[:8]}",
                "ip_address": self._generate_ip_address(is_suspicious=True),
                "email_domain": customer["email_domain"],
                "billing_country": customer["home_country"],
                "shipping_country": customer["home_country"],
                "is_online": True,
                "product_category": profile.category,
                "data_source": "SYNTHETIC",
                "is_fraud": True,
                "fraud_label_source": "INJECTED",
            }
            transactions.append(txn)

        # Duplicates
        for _ in range(anomaly_counts["duplicate"]):
            if transactions:
                base_txn = np.random.choice(
                    [t for t in transactions if not t.get("is_fraud", False)]
                )
                transactions.append(self._inject_duplicate(base_txn))

        # Merchant anomalies
        for _ in range(anomaly_counts["merchant"]):
            customer = np.random.choice(self._customer_pool)
            base_date = start_date + timedelta(days=np.random.randint(0, days_range))
            profile = np.random.choice(self.MERCHANT_PROFILES)

            base_txn = {
                "transaction_id": f"txn_{uuid.uuid4().hex}",
                "timestamp": self._generate_transaction_time(base_date, is_anomaly=True),
                "amount": round(self._generate_amount(profile), 2),
                "currency": "USD",
                "merchant_id": np.random.choice(
                    self._merchant_pool.get(profile.category, ["unknown"])
                ),
                "merchant_category": profile.category,
                "card_id": customer["primary_card"],
                "card_type": np.random.choice(self.CARD_TYPES),
                "card_country": customer["home_country"],
                "customer_id": customer["customer_id"],
                "device_id": f"device_{uuid.uuid4().hex[:8]}",
                "ip_address": self._generate_ip_address(),
                "email_domain": customer["email_domain"],
                "billing_country": customer["home_country"],
                "shipping_country": customer["home_country"],
                "is_online": True,
                "product_category": profile.category,
                "data_source": "SYNTHETIC",
                "is_fraud": False,
                "fraud_label_source": "INJECTED",
            }
            transactions.append(self._inject_merchant_anomaly(base_txn))

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Log statistics
        fraud_rate = df["is_fraud"].mean()
        logger.info(
            "Synthetic data generation complete",
            total_transactions=len(df),
            fraud_transactions=df["is_fraud"].sum(),
            fraud_rate=f"{fraud_rate:.2%}",
            unique_cards=df["card_id"].nunique(),
            unique_merchants=df["merchant_id"].nunique(),
            date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        )

        return df

    def save_to_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Save DataFrame to Parquet format."""
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved to Parquet", path=path, rows=len(df))

    def save_to_csv(self, df: pd.DataFrame, path: str) -> None:
        """Save DataFrame to CSV format."""
        df.to_csv(path, index=False)
        logger.info("Saved to CSV", path=path, rows=len(df))
