"""IEEE-CIS Fraud Detection dataset loader.

Dataset: https://www.kaggle.com/c/ieee-fraud-detection
Paper: "Fraud Detection Using IEEE-CIS Dataset with Machine Learning Algorithms"

The IEEE-CIS dataset contains real-world e-commerce transaction data with:
- 590,540 transactions (train) + 506,691 (test)
- 394 features across transaction and identity tables
- ~3.5% fraud rate in training set

This loader handles:
- Downloading from Kaggle (requires kaggle.json credentials)
- Memory-efficient loading with chunking
- Feature type mapping to our schema
- Train/test split preservation
"""

from typing import Any

import pandas as pd

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class IEEECISLoader:
    """
    Load and preprocess IEEE-CIS Fraud Detection dataset.

    Usage:
        loader = IEEECISLoader()

        # Download from Kaggle (requires credentials)
        loader.download()

        # Load and transform to our schema
        df = loader.load_transactions()
    """

    KAGGLE_COMPETITION = "ieee-fraud-detection"

    # Mapping from IEEE-CIS columns to our schema
    COLUMN_MAPPING = {
        "TransactionID": "transaction_id",
        "TransactionDT": "transaction_dt",  # Seconds from reference
        "TransactionAmt": "amount",
        "ProductCD": "product_category",
        "card1": "card_id",  # Hashed card identifier
        "card4": "card_type",  # Visa, Mastercard, etc.
        "card6": "card_type_detail",  # Credit, debit
        "addr1": "billing_region",
        "addr2": "billing_country",
        "P_emaildomain": "email_domain",
        "R_emaildomain": "recipient_email_domain",
        "DeviceType": "device_type",
        "DeviceInfo": "device_info",
        "isFraud": "is_fraud",
    }

    # Card type mapping
    CARD_TYPE_MAPPING = {
        "visa": "VISA",
        "mastercard": "MASTERCARD",
        "american express": "AMEX",
        "discover": "DISCOVER",
    }

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize loader."""
        self.settings = settings or get_settings()
        self.data_dir = self.settings.data_dir / "ieee-cis"

    def download(self, force: bool = False) -> None:
        """
        Download dataset from Kaggle.

        Requires:
            - kaggle.json in ~/.kaggle/ with API credentials
            - Acceptance of competition rules on Kaggle

        Args:
            force: Re-download even if files exist.
        """
        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "kaggle package required. Install with: pip install kaggle\n"
                "Then place kaggle.json in ~/.kaggle/"
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)

        expected_files = [
            "train_transaction.csv",
            "train_identity.csv",
            "test_transaction.csv",
            "test_identity.csv",
        ]

        if not force and all((self.data_dir / f).exists() for f in expected_files):
            logger.info("Dataset already downloaded", path=str(self.data_dir))
            return

        logger.info("Downloading IEEE-CIS dataset from Kaggle...")

        kaggle.api.competition_download_files(
            self.KAGGLE_COMPETITION,
            path=str(self.data_dir),
            quiet=False,
        )

        # Unzip if needed
        zip_path = self.data_dir / f"{self.KAGGLE_COMPETITION}.zip"
        if zip_path.exists():
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
            zip_path.unlink()

        logger.info("Download complete", path=str(self.data_dir))

    def load_transactions(
        self,
        include_identity: bool = True,
        include_test: bool = False,
        sample_frac: float | None = None,
        chunksize: int | None = None,
    ) -> pd.DataFrame:
        """
        Load IEEE-CIS transactions and transform to our schema.

        Args:
            include_identity: Merge identity features.
            include_test: Include test set (no labels).
            sample_frac: Random sample fraction (0-1).
            chunksize: If set, return iterator of chunks.

        Returns:
            DataFrame with transactions in our schema.
        """
        # Load transaction data
        train_txn_path = self.data_dir / "train_transaction.csv"

        if not train_txn_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {train_txn_path}. "
                f"Run loader.download() first or download manually from Kaggle."
            )

        logger.info("Loading IEEE-CIS transactions", path=str(train_txn_path))

        # Select columns to reduce memory
        usecols = list(self.COLUMN_MAPPING.keys())

        # Load transactions
        df = pd.read_csv(
            train_txn_path,
            usecols=lambda x: x in usecols or x.startswith(("C", "D", "V", "M")),
            dtype={"card1": str, "card4": str},
        )

        logger.info("Loaded transactions", rows=len(df), columns=len(df.columns))

        # Merge identity features if requested
        if include_identity:
            identity_path = self.data_dir / "train_identity.csv"
            if identity_path.exists():
                identity_df = pd.read_csv(identity_path)
                df = df.merge(identity_df, on="TransactionID", how="left")
                logger.info("Merged identity features", identity_rows=len(identity_df))

        # Transform to our schema
        df = self._transform_schema(df)

        # Sample if requested
        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            logger.info("Sampled dataset", rows=len(df), frac=sample_frac)

        return df

    def _transform_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform IEEE-CIS schema to our standard schema."""
        result = pd.DataFrame()

        # Map basic columns
        result["transaction_id"] = "ieee_" + df["TransactionID"].astype(str)

        # Convert TransactionDT (seconds from reference) to timestamp
        # Reference appears to be around 2017-11-30
        reference_date = pd.Timestamp("2017-11-30")
        result["timestamp"] = reference_date + pd.to_timedelta(df["TransactionDT"], unit="s")

        result["amount"] = df["TransactionAmt"].round(2)
        result["currency"] = "USD"

        # Card info
        result["card_id"] = "ieee_card_" + df["card1"].astype(str)
        result["card_type"] = df["card4"].str.lower().map(self.CARD_TYPE_MAPPING).fillna("UNKNOWN")

        # Get billing country from addr2 (most common values are 87.0 = US)
        if "addr2" in df.columns:
            result["billing_country"] = df["addr2"].apply(
                lambda x: "USA" if x == 87.0 else ("CAN" if x == 60.0 else "OTHER")
            )
        else:
            result["billing_country"] = "USA"

        result["card_country"] = result["billing_country"]
        result["shipping_country"] = result["billing_country"]

        # Merchant info (not directly available, use ProductCD)
        result["merchant_id"] = "ieee_merchant_" + df["ProductCD"].astype(str)
        result["merchant_category"] = (
            df["ProductCD"]
            .map(
                {
                    "W": "digital_goods",
                    "H": "home",
                    "C": "clothing",
                    "S": "services",
                    "R": "retail",
                }
            )
            .fillna("unknown")
        )

        # Email domain
        if "P_emaildomain" in df.columns:
            result["email_domain"] = df["P_emaildomain"].fillna("unknown")
        else:
            result["email_domain"] = "unknown"

        # Device info
        if "DeviceType" in df.columns:
            result["is_online"] = True  # All IEEE-CIS transactions are online
            result["device_id"] = "ieee_device_" + df.get("DeviceInfo", "unknown").astype(str)
        else:
            result["is_online"] = True
            result["device_id"] = "unknown"

        # Customer (approximation from card)
        result["customer_id"] = result["card_id"]

        # IP address not directly available
        result["ip_address"] = None

        # Product category
        result["product_category"] = result["merchant_category"]

        # Source and labels
        result["data_source"] = "IEEE_CIS"
        result["is_fraud"] = df["isFraud"].astype(bool)
        result["fraud_label_source"] = "GROUND_TRUTH"

        # Keep original V-features for modeling (velocity, count features)
        v_cols = [c for c in df.columns if c.startswith("V")]
        c_cols = [c for c in df.columns if c.startswith("C") and c != "card"]
        d_cols = [c for c in df.columns if c.startswith("D")]

        for col in v_cols + c_cols + d_cols:
            if col in df.columns:
                result[f"ieee_{col.lower()}"] = df[col]

        logger.info(
            "Transformed to standard schema",
            output_columns=len(result.columns),
            fraud_rate=f"{result['is_fraud'].mean():.2%}",
        )

        return result

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split preserving fraud ratio.

        Args:
            test_size: Fraction for test set.
            stratify: Maintain fraud class balance.

        Returns:
            Tuple of (train_df, test_df).
        """
        from sklearn.model_selection import train_test_split

        df = self.load_transactions()

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["is_fraud"] if stratify else None,
            random_state=42,
        )

        logger.info(
            "Created train/test split",
            train_size=len(train_df),
            test_size=len(test_df),
            train_fraud_rate=f"{train_df['is_fraud'].mean():.2%}",
            test_fraud_rate=f"{test_df['is_fraud'].mean():.2%}",
        )

        return train_df, test_df

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        df = self.load_transactions(include_identity=False, sample_frac=0.1)

        return {
            "total_transactions": len(df) * 10,  # Extrapolate from sample
            "fraud_rate": float(df["is_fraud"].mean()),
            "unique_cards": df["card_id"].nunique() * 10,
            "date_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
            "amount_stats": {
                "mean": float(df["amount"].mean()),
                "median": float(df["amount"].median()),
                "std": float(df["amount"].std()),
                "min": float(df["amount"].min()),
                "max": float(df["amount"].max()),
            },
            "merchant_categories": df["merchant_category"].value_counts().to_dict(),
        }
