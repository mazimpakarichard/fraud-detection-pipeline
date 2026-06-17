# Fraud Detection Pipeline

Production-grade transaction anomaly and fraud detection pipeline with ensemble ML models, explainability, and full MLOps infrastructure.

## Features

- **Multi-model Ensemble**: Rule-based + Isolation Forest + PyTorch Autoencoder
- **Explainability**: Per-transaction reason codes for flagged transactions
- **Data Sources**: IEEE-CIS dataset support + synthetic data generator (>1M rows)
- **Orchestration**: Apache Airflow DAG for automated batch processing
- **Data Quality**: Great Expectations validation suite
- **Monitoring**: Evidently AI drift detection and performance tracking
- **Security**: Secrets via environment variables, audit logging

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)

### Installation

```bash
# Clone repository
git clone https://github.com/example/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with development dependencies
pip install -e ".[dev]"
```

### Running with Docker

```bash
# Start all services (PostgreSQL, Airflow, Scoring Service)
docker compose up -d

# Initialize Airflow
docker compose run airflow-init

# Access services:
# - Airflow UI: http://localhost:8080 (admin/admin)
# - Scoring API: http://localhost:8000
# - PostgreSQL: localhost:5432
```

### Local Development

```bash
# Set environment variables
export FRAUD_DB_HOST=localhost
export FRAUD_DB_PORT=5432
export FRAUD_DB_NAME=fraud_detection
export FRAUD_DB_USER=fraud
export FRAUD_DB_PASSWORD=your_password

# Initialize database schema
psql -h $FRAUD_DB_HOST -U $FRAUD_DB_USER -d $FRAUD_DB_NAME -f sql/001_schema.sql

# Run synthetic data generation and scoring
python -m fraud_detection.cli generate --rows 100000
python -m fraud_detection.cli train
python -m fraud_detection.cli score --batch-size 10000
```

## Project Structure

```
fraud-detection-pipeline/
├── src/fraud_detection/
│   ├── data/               # Data loaders (synthetic, IEEE-CIS)
│   ├── features/           # Feature engineering
│   ├── models/             # ML models (rules, IF, autoencoder, ensemble)
│   ├── monitoring/         # Drift detection, audit logging
│   ├── validation/         # Great Expectations validators
│   ├── utils/              # Config, database, logging
│   └── cli.py              # Command-line interface
├── dags/                   # Airflow DAGs
├── sql/                    # Database schema and seeds
├── tests/                  # Unit and integration tests
├── docs/                   # Additional documentation
└── docker-compose.yml      # Service orchestration
```

## Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Extract   │────▶│  Validate   │────▶│  Transform  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Monitor   │◀────│   Persist   │◀────│    Score    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│    Drift    │     │    Alert    │
│   Reports   │     │   System    │
└─────────────┘     └─────────────┘
```

## Models

### Rule-Based Model
Deterministic rules based on domain knowledge:
- High-value transaction detection (>$5000)
- Velocity anomalies (>5 transactions/hour)
- Geographic inconsistencies
- Suspicious email domains
- Off-hours activity patterns

### Isolation Forest
Unsupervised anomaly detection:
- 200 estimators
- Auto contamination rate
- Feature importance via path length

### PyTorch Autoencoder
Deep learning reconstruction-based detection:
- Symmetric encoder-decoder
- Bottleneck dimension: 16
- Reconstruction error as anomaly score

### Ensemble Scoring
Weighted combination (configurable):
- Rules: 30%
- Isolation Forest: 35%
- Autoencoder: 35%

## Explainability

Each flagged transaction includes reason codes:

```json
{
  "transaction_id": "txn_abc123",
  "ensemble_score": 0.85,
  "is_flagged": true,
  "reason_codes": [
    {
      "model": "rules",
      "rule": "high_amount",
      "score": 0.95,
      "reason": "Amount $9,999.99 exceeds threshold $5,000.00"
    },
    {
      "model": "isolation_forest",
      "feature": "card_txn_count_1h",
      "contribution": 0.42,
      "reason": "Feature 'card_txn_count_1h' contributed to anomaly score"
    }
  ]
}
```

## Configuration

Environment variables (prefix: `FRAUD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_DB_HOST` | localhost | PostgreSQL host |
| `FRAUD_DB_PORT` | 5432 | PostgreSQL port |
| `FRAUD_DB_NAME` | fraud_detection | Database name |
| `FRAUD_DB_USER` | postgres | Database user |
| `FRAUD_DB_PASSWORD` | - | Database password (required) |
| `FRAUD_USE_SYNTHETIC_DATA` | true | Use synthetic vs IEEE-CIS |
| `FRAUD_SYNTHETIC_ROWS` | 1000000 | Synthetic data size |
| `FRAUD_ANOMALY_RATE` | 0.02 | Synthetic anomaly rate |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fraud_detection --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Monitoring

### Data Drift Detection
Evidently AI monitors feature distributions:
- Statistical tests for drift detection
- HTML reports in `reports/` directory
- Configurable drift thresholds

### Audit Logging
All pipeline operations logged to `fraud.audit_log`:
- Operation type (EXTRACT, VALIDATE, SCORE, etc.)
- Duration and record counts
- Error messages on failure
- Triggered by user/system

## Security

- Credentials via environment variables only
- SecretStr for sensitive values
- No credentials in code or logs
- SQL injection prevention via parameterized queries
- Audit trail for compliance

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and linting
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
