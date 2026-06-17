# Data Lineage Documentation

## Overview

This document describes the data flow, transformations, and governance for the Fraud Detection Pipeline.

## Data Sources

### 1. Synthetic Transaction Generator

**Source**: `fraud_detection.data.synthetic.SyntheticTransactionGenerator`

**Description**: Generates realistic synthetic transaction data with configurable anomaly injection for development and testing.

**Data Characteristics**:
- Volume: 1M+ transactions (configurable)
- Anomaly Rate: 2% (configurable)
- Time Range: 90 days rolling window

**Anomaly Types Injected**:
| Type | Ratio | Description |
|------|-------|-------------|
| Amount Outliers | 25% | 10-50x normal transaction max |
| Velocity Anomalies | 20% | 5+ transactions within minutes |
| Geographic Anomalies | 15% | Impossible travel patterns |
| Off-Hours Activity | 15% | Transactions 1-5 AM |
| Duplicates | 10% | Near-identical transactions |
| Merchant Anomalies | 15% | Unusual category for card |

**Schema**:
```
transaction_id: VARCHAR(64) - Unique identifier
timestamp: TIMESTAMP WITH TIME ZONE - Transaction time
amount: DECIMAL(18,2) - Transaction amount
currency: VARCHAR(3) - Currency code
merchant_id: VARCHAR(64) - Merchant identifier
merchant_category: VARCHAR(100) - MCC category
card_id: VARCHAR(64) - Hashed card identifier
card_type: VARCHAR(20) - VISA, MASTERCARD, etc.
card_country: VARCHAR(3) - Card issuing country
customer_id: VARCHAR(64) - Customer identifier
device_id: VARCHAR(64) - Device fingerprint
ip_address: VARCHAR(45) - Transaction IP
email_domain: VARCHAR(255) - Email domain
billing_country: VARCHAR(3) - Billing address country
shipping_country: VARCHAR(3) - Shipping address country
is_online: BOOLEAN - Online vs in-store
product_category: VARCHAR(100) - Product type
data_source: VARCHAR(20) - 'SYNTHETIC'
is_fraud: BOOLEAN - Fraud label
fraud_label_source: VARCHAR(50) - 'INJECTED'
```

### 2. IEEE-CIS Fraud Detection Dataset

**Source**: Kaggle Competition `ieee-fraud-detection`

**Description**: Real-world e-commerce transaction data from IEEE Computational Intelligence Society.

**Data Characteristics**:
- Volume: 590,540 training transactions
- Fraud Rate: ~3.5%
- Features: 394 original columns

**Column Mapping**:
| IEEE-CIS Column | Our Schema | Notes |
|-----------------|------------|-------|
| TransactionID | transaction_id | Prefixed with 'ieee_' |
| TransactionDT | timestamp | Converted from seconds |
| TransactionAmt | amount | Rounded to 2 decimals |
| ProductCD | merchant_category | Mapped to categories |
| card1-6 | card_id, card_type | Hashed identifiers |
| addr1-2 | billing_country | Mapped to ISO codes |
| P_emaildomain | email_domain | Direct mapping |
| isFraud | is_fraud | Ground truth label |

**Data Access**:
- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Must accept competition rules
- Download via `IEEECISLoader.download()`

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                            │
│  ┌─────────────────┐         ┌─────────────────┐             │
│  │   Synthetic     │         │    IEEE-CIS     │             │
│  │   Generator     │         │    Dataset      │             │
│  └────────┬────────┘         └────────┬────────┘             │
└───────────┼───────────────────────────┼──────────────────────┘
            │                           │
            ▼                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    EXTRACTION LAYER                           │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  fraud.transactions table (PostgreSQL)               │     │
│  │  - Raw transaction records                           │     │
│  │  - Source tracking (data_source column)              │     │
│  │  - Label tracking (fraud_label_source column)        │     │
│  └───────────────────────────┬─────────────────────────┘     │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   VALIDATION LAYER                            │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Great Expectations Suite                            │     │
│  │  - Column presence validation                        │     │
│  │  - Null checks                                       │     │
│  │  - Range validation                                  │     │
│  │  - Uniqueness constraints                            │     │
│  └───────────────────────────┬─────────────────────────┘     │
│                              │                                │
│  Results stored in: fraud.validation_results                 │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 TRANSFORMATION LAYER                          │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Feature Engineering                                 │     │
│  │                                                      │     │
│  │  Temporal Features:                                  │     │
│  │  - hour_sin, hour_cos (cyclical)                     │     │
│  │  - dow_sin, dow_cos (cyclical)                       │     │
│  │  - is_weekend, is_night, is_business_hours           │     │
│  │                                                      │     │
│  │  Risk Indicators:                                    │     │
│  │  - suspicious_email                                  │     │
│  │  - country_mismatch                                  │     │
│  │  - high_risk_category                                │     │
│  │  - is_large_amount                                   │     │
│  │                                                      │     │
│  │  Velocity Features (per entity):                     │     │
│  │  - card_hours_since_last                             │     │
│  │  - card_txn_count_1h, _24h, _7d                      │     │
│  │  - card_amount_mean_24h, _std_24h                    │     │
│  │  - card_amount_zscore_24h                            │     │
│  │  - card_unique_merchants_24h                         │     │
│  └───────────────────────────┬─────────────────────────┘     │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    SCORING LAYER                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Ensemble Model                                      │     │
│  │  - Rule-based scores + reasons                       │     │
│  │  - Isolation Forest anomaly scores                   │     │
│  │  - Autoencoder reconstruction errors                 │     │
│  │  - Weighted ensemble combination                     │     │
│  │  - Explainability reason codes                       │     │
│  └───────────────────────────┬─────────────────────────┘     │
│                              │                                │
│  Results stored in: fraud.scoring_results                    │
│  - ensemble_score, rule_score, if_score, ae_score            │
│  - is_flagged, reason_codes (JSONB)                          │
│  - model_versions (JSONB)                                    │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   MONITORING LAYER                            │
│  ┌─────────────────┐         ┌─────────────────┐             │
│  │   Drift         │         │   Performance   │             │
│  │   Detection     │         │   Metrics       │             │
│  │   (Evidently)   │         │   Tracking      │             │
│  └────────┬────────┘         └────────┬────────┘             │
│           │                           │                       │
│  Results: fraud.drift_reports        fraud.audit_log         │
└───────────┴───────────────────────────┴──────────────────────┘
```

## Data Quality Rules

### Validation Suite

| Expectation | Column | Rule |
|-------------|--------|------|
| Not Null | transaction_id | Required |
| Unique | transaction_id | No duplicates |
| Not Null | amount | Required |
| Between | amount | 0 - 1,000,000 |
| Not Null | timestamp | Required |
| Not Null | card_id | Required |
| Not Null | merchant_id | Required |

### Data Quality Metrics

Tracked per batch:
- Null rate by column
- Duplicate rate
- Schema compliance
- Value distribution stats

## Data Retention

| Data Type | Retention | Location |
|-----------|-----------|----------|
| Raw Transactions | 2 years | fraud.transactions |
| Scoring Results | 2 years | fraud.scoring_results |
| Audit Logs | 7 years | fraud.audit_log |
| Validation Results | 90 days | fraud.validation_results |
| Drift Reports | 1 year | fraud.drift_reports |
| Feature Store | 30 days | fraud.feature_store |

## Access Control

### Database Roles
- `fraud_reader`: SELECT on all tables
- `fraud_writer`: INSERT/UPDATE on transactions, scoring_results
- `fraud_admin`: Full access including schema changes
- `fraud_pipeline`: Application service account

### Audit Requirements
All data access logged to `fraud.audit_log`:
- Timestamp
- Operation type
- Records affected
- Triggered by (user/system)

## Compliance

### GDPR Considerations
- No direct PII stored (card_id is hashed)
- Right to erasure via transaction_id
- Data minimization principles applied

### PCI-DSS
- Card numbers never stored
- Hashed identifiers only
- Encrypted connections required

## Change History

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2024-01 | 1.0 | Initial documentation | Team |
