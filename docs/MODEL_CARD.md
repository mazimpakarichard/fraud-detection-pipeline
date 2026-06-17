# Model Card: Fraud Detection Ensemble

## Model Details

### Overview
- **Model Name**: Fraud Detection Ensemble v1.0.0
- **Model Type**: Anomaly Detection (Unsupervised + Rules)
- **Task**: Transaction fraud detection
- **Date**: 2024
- **License**: MIT

### Model Architecture
Ensemble of three complementary approaches:

1. **Rule-Based Model (30% weight)**
   - Deterministic domain rules
   - Interpretable thresholds
   - Real-time scoring capable

2. **Isolation Forest (35% weight)**
   - 200 decision trees
   - Auto contamination estimation
   - Anomaly = fewer splits to isolate

3. **PyTorch Autoencoder (35% weight)**
   - Architecture: 64-32-16-32-64
   - Batch normalization + dropout (0.1)
   - Reconstruction error as anomaly score

## Intended Use

### Primary Use Cases
- Batch scoring of e-commerce transactions
- Real-time fraud screening (rule-based component)
- Alert generation for manual review
- Risk assessment and prioritization

### Out-of-Scope Uses
- Automated transaction blocking (requires human review)
- Credit scoring or lending decisions
- Personal identification or tracking
- Use outside financial fraud domain

## Training Data

### Data Sources
1. **Synthetic Data Generator**
   - 1M+ transactions
   - Configurable anomaly rate (default 2%)
   - Injected patterns: velocity, amount outliers, geographic

2. **IEEE-CIS Fraud Detection Dataset**
   - 590,540 labeled transactions
   - 3.5% fraud rate
   - Real e-commerce transactions

### Data Characteristics
| Metric | Value |
|--------|-------|
| Training samples | 1,000,000 |
| Features | 25-50 engineered |
| Fraud rate | 2-3.5% |
| Time span | 90 days |

## Performance Metrics

### Evaluation Methodology
- Stratified train/test split (80/20)
- Cross-validation for hyperparameter tuning
- Held-out test set for final evaluation

### Expected Performance
| Metric | Synthetic | IEEE-CIS |
|--------|-----------|----------|
| Precision | 0.85-0.90 | 0.75-0.85 |
| Recall | 0.75-0.85 | 0.70-0.80 |
| F1 Score | 0.80-0.87 | 0.72-0.82 |
| ROC-AUC | 0.92-0.96 | 0.88-0.94 |
| PR-AUC | 0.75-0.85 | 0.65-0.78 |

### Performance by Subgroup
Performance may vary by:
- Transaction amount (higher amounts = better detection)
- Merchant category (digital goods vs physical)
- Time of day (night transactions = more false positives)

## Limitations and Risks

### Known Limitations
1. **Concept Drift**: Performance degrades as fraud patterns evolve
2. **Class Imbalance**: 2-3% fraud rate limits recall
3. **Feature Availability**: Requires all expected features
4. **Cold Start**: New cards/merchants lack history

### Potential Biases
- Geographic bias: Trained primarily on US transactions
- Temporal bias: Patterns may not generalize across seasons
- Merchant bias: May flag legitimate high-value purchases

### Mitigation Strategies
- Continuous monitoring with Evidently AI
- Regular retraining (weekly/monthly)
- Human-in-the-loop for final decisions
- Explainability for flagged transactions

## Ethical Considerations

### Fairness
- Model does not use protected attributes (race, gender, etc.)
- Regular fairness audits recommended
- False positive impact on legitimate users considered

### Privacy
- No PII stored in model
- Card IDs are hashed/anonymized
- Audit logs for data access

### Accountability
- Full audit trail of model decisions
- Reason codes for all flagged transactions
- Model version tracking in predictions

## Model Maintenance

### Monitoring
- Feature drift detection (Evidently)
- Performance monitoring (precision/recall over time)
- Data quality validation (Great Expectations)

### Update Frequency
- **Retraining**: Monthly or when drift detected
- **Rules Update**: As needed based on new patterns
- **Full Retrain**: Quarterly with new data

### Rollback Procedure
1. Model versions stored in `models/` directory
2. Ensemble config includes version metadata
3. Database tracks model version per prediction

## Contact

- **Team**: Fraud Detection Engineering
- **Email**: fraud-team@example.com
- **Repository**: github.com/example/fraud-detection-pipeline

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01 | Initial release |
