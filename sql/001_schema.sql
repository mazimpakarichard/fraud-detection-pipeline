-- ============================================================================
-- Fraud Detection Pipeline - Database Schema
-- ============================================================================
-- Performance-optimized schema for high-volume transaction processing
-- Includes proper indexes, partitioning considerations, and audit tables
-- ============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS fraud;

-- Set search path
SET search_path TO fraud, public;

-- ============================================================================
-- CORE TRANSACTION TABLES
-- ============================================================================

-- Main transactions table
-- Designed for high-volume inserts and efficient time-range queries
CREATE TABLE IF NOT EXISTS fraud.transactions (
    transaction_id      VARCHAR(64) PRIMARY KEY,
    timestamp           TIMESTAMP WITH TIME ZONE NOT NULL,
    amount              DECIMAL(18, 2) NOT NULL,
    currency            VARCHAR(3) DEFAULT 'USD',
    merchant_id         VARCHAR(64) NOT NULL,
    merchant_category   VARCHAR(100),
    card_id             VARCHAR(64) NOT NULL,
    card_type           VARCHAR(20),
    card_country        VARCHAR(3),
    customer_id         VARCHAR(64),
    device_id           VARCHAR(64),
    ip_address          VARCHAR(45),  -- Supports IPv6
    email_domain        VARCHAR(255),
    billing_country     VARCHAR(3),
    shipping_country    VARCHAR(3),
    is_online           BOOLEAN DEFAULT TRUE,
    product_category    VARCHAR(100),
    -- Metadata
    data_source         VARCHAR(20) NOT NULL DEFAULT 'UNKNOWN',  -- 'SYNTHETIC', 'IEEE_CIS', 'PRODUCTION'
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- For labeled data (training/evaluation)
    is_fraud            BOOLEAN,
    fraud_label_source  VARCHAR(50)  -- 'GROUND_TRUTH', 'INJECTED', 'UNKNOWN'
);

-- Performance indexes for transactions
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
    ON fraud.transactions (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_transactions_card_id
    ON fraud.transactions (card_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id
    ON fraud.transactions (merchant_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_id
    ON fraud.transactions (customer_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_transactions_amount
    ON fraud.transactions (amount);

CREATE INDEX IF NOT EXISTS idx_transactions_data_source
    ON fraud.transactions (data_source, timestamp DESC);

-- Composite index for common fraud detection queries
CREATE INDEX IF NOT EXISTS idx_transactions_fraud_analysis
    ON fraud.transactions (card_id, merchant_id, timestamp DESC, amount);

-- ============================================================================
-- SCORING RESULTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fraud.scoring_results (
    result_id           SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64) NOT NULL REFERENCES fraud.transactions(transaction_id),
    batch_id            VARCHAR(64) NOT NULL,
    scored_at           TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Individual model scores (0-1, higher = more anomalous)
    rule_score          DECIMAL(5, 4),
    isolation_forest_score DECIMAL(5, 4),
    autoencoder_score   DECIMAL(5, 4),

    -- Ensemble score
    ensemble_score      DECIMAL(5, 4) NOT NULL,

    -- Binary prediction
    is_flagged          BOOLEAN NOT NULL DEFAULT FALSE,
    flag_threshold      DECIMAL(5, 4),

    -- Explainability: top contributing factors (JSON array)
    reason_codes        JSONB,

    -- Model versions used
    model_versions      JSONB,

    CONSTRAINT unique_transaction_batch UNIQUE (transaction_id, batch_id)
);

-- Indexes for scoring results
CREATE INDEX IF NOT EXISTS idx_scoring_results_batch
    ON fraud.scoring_results (batch_id, scored_at DESC);

CREATE INDEX IF NOT EXISTS idx_scoring_results_flagged
    ON fraud.scoring_results (is_flagged, scored_at DESC)
    WHERE is_flagged = TRUE;

CREATE INDEX IF NOT EXISTS idx_scoring_results_ensemble
    ON fraud.scoring_results (ensemble_score DESC);

CREATE INDEX IF NOT EXISTS idx_scoring_results_transaction
    ON fraud.scoring_results (transaction_id);

-- ============================================================================
-- AUDIT LOG TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fraud.audit_log (
    audit_id            SERIAL PRIMARY KEY,
    timestamp           TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    batch_id            VARCHAR(64) NOT NULL,
    operation           VARCHAR(50) NOT NULL,  -- 'EXTRACT', 'VALIDATE', 'SCORE', 'PERSIST', 'ALERT'
    status              VARCHAR(20) NOT NULL,  -- 'STARTED', 'COMPLETED', 'FAILED'
    records_processed   INTEGER,
    records_flagged     INTEGER,
    duration_seconds    DECIMAL(10, 3),
    error_message       TEXT,
    metadata            JSONB,
    -- Security: who/what triggered this
    triggered_by        VARCHAR(100),
    source_system       VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_audit_log_batch
    ON fraud.audit_log (batch_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_status
    ON fraud.audit_log (status, timestamp DESC);

-- ============================================================================
-- DATA VALIDATION RESULTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS fraud.validation_results (
    validation_id       SERIAL PRIMARY KEY,
    batch_id            VARCHAR(64) NOT NULL,
    validated_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expectation_suite   VARCHAR(100) NOT NULL,
    success             BOOLEAN NOT NULL,
    statistics          JSONB,
    failed_expectations JSONB,
    metadata            JSONB
);

CREATE INDEX IF NOT EXISTS idx_validation_batch
    ON fraud.validation_results (batch_id, validated_at DESC);

-- ============================================================================
-- MONITORING / DRIFT REPORTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS fraud.drift_reports (
    report_id           SERIAL PRIMARY KEY,
    batch_id            VARCHAR(64) NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    report_type         VARCHAR(50) NOT NULL,  -- 'DATA_DRIFT', 'MODEL_PERFORMANCE', 'FEATURE_DRIFT'
    reference_period    TSTZRANGE,
    current_period      TSTZRANGE,
    drift_detected      BOOLEAN,
    drift_score         DECIMAL(5, 4),
    metrics             JSONB,
    report_html_path    TEXT,
    report_json_path    TEXT
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_batch
    ON fraud.drift_reports (batch_id, created_at DESC);

-- ============================================================================
-- FEATURE STORE (MATERIALIZED FEATURES)
-- ============================================================================

CREATE TABLE IF NOT EXISTS fraud.feature_store (
    feature_id          SERIAL PRIMARY KEY,
    entity_type         VARCHAR(20) NOT NULL,  -- 'CARD', 'MERCHANT', 'CUSTOMER'
    entity_id           VARCHAR(64) NOT NULL,
    feature_window      VARCHAR(20) NOT NULL,  -- '1H', '24H', '7D', '30D'
    computed_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    valid_until         TIMESTAMP WITH TIME ZONE,
    features            JSONB NOT NULL,

    CONSTRAINT unique_entity_feature UNIQUE (entity_type, entity_id, feature_window)
);

CREATE INDEX IF NOT EXISTS idx_feature_store_entity
    ON fraud.feature_store (entity_type, entity_id, computed_at DESC);

CREATE INDEX IF NOT EXISTS idx_feature_store_validity
    ON fraud.feature_store (valid_until)
    WHERE valid_until IS NOT NULL;

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- View: Recent flagged transactions with details
CREATE OR REPLACE VIEW fraud.v_flagged_transactions AS
SELECT
    t.transaction_id,
    t.timestamp,
    t.amount,
    t.merchant_id,
    t.merchant_category,
    t.card_id,
    t.is_online,
    sr.ensemble_score,
    sr.rule_score,
    sr.isolation_forest_score,
    sr.autoencoder_score,
    sr.reason_codes,
    sr.scored_at,
    sr.batch_id,
    t.is_fraud AS ground_truth
FROM fraud.transactions t
JOIN fraud.scoring_results sr ON t.transaction_id = sr.transaction_id
WHERE sr.is_flagged = TRUE
ORDER BY sr.scored_at DESC;

-- View: Batch processing summary
CREATE OR REPLACE VIEW fraud.v_batch_summary AS
SELECT
    batch_id,
    MIN(scored_at) AS batch_start,
    MAX(scored_at) AS batch_end,
    COUNT(*) AS total_scored,
    SUM(CASE WHEN is_flagged THEN 1 ELSE 0 END) AS total_flagged,
    AVG(ensemble_score) AS avg_score,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ensemble_score) AS median_score,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ensemble_score) AS p95_score
FROM fraud.scoring_results
GROUP BY batch_id
ORDER BY batch_start DESC;

-- ============================================================================
-- FUNCTIONS FOR EFFICIENT DATA ACCESS
-- ============================================================================

-- Function: Get velocity features for a card (window functions)
CREATE OR REPLACE FUNCTION fraud.get_card_velocity(
    p_card_id VARCHAR(64),
    p_reference_time TIMESTAMP WITH TIME ZONE,
    p_window_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    txn_count_window BIGINT,
    total_amount_window DECIMAL(18, 2),
    avg_amount_window DECIMAL(18, 2),
    distinct_merchants BIGINT,
    distinct_countries BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT,
        COALESCE(SUM(amount), 0)::DECIMAL(18, 2),
        COALESCE(AVG(amount), 0)::DECIMAL(18, 2),
        COUNT(DISTINCT merchant_id)::BIGINT,
        COUNT(DISTINCT billing_country)::BIGINT
    FROM fraud.transactions
    WHERE card_id = p_card_id
      AND timestamp >= p_reference_time - (p_window_hours || ' hours')::INTERVAL
      AND timestamp < p_reference_time;
END;
$$ LANGUAGE plpgsql;

-- Function: Extract batch for scoring (efficient pagination)
CREATE OR REPLACE FUNCTION fraud.extract_unscored_batch(
    p_batch_size INTEGER DEFAULT 10000,
    p_data_source VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    transaction_id VARCHAR(64),
    timestamp TIMESTAMP WITH TIME ZONE,
    amount DECIMAL(18, 2),
    merchant_id VARCHAR(64),
    merchant_category VARCHAR(100),
    card_id VARCHAR(64),
    card_type VARCHAR(20),
    is_online BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.transaction_id,
        t.timestamp,
        t.amount,
        t.merchant_id,
        t.merchant_category,
        t.card_id,
        t.card_type,
        t.is_online
    FROM fraud.transactions t
    LEFT JOIN fraud.scoring_results sr ON t.transaction_id = sr.transaction_id
    WHERE sr.transaction_id IS NULL
      AND (p_data_source IS NULL OR t.data_source = p_data_source)
    ORDER BY t.timestamp
    LIMIT p_batch_size;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE fraud.transactions IS 'Core transaction data from all sources (synthetic, IEEE-CIS, production)';
COMMENT ON TABLE fraud.scoring_results IS 'Fraud scoring results with ensemble scores and explainability';
COMMENT ON TABLE fraud.audit_log IS 'Immutable audit trail for all pipeline operations';
COMMENT ON TABLE fraud.validation_results IS 'Data validation results from Great Expectations';
COMMENT ON TABLE fraud.drift_reports IS 'Data drift and model performance monitoring reports';
COMMENT ON TABLE fraud.feature_store IS 'Pre-computed features for entities (cards, merchants, customers)';

COMMENT ON COLUMN fraud.transactions.data_source IS 'Origin of data: SYNTHETIC, IEEE_CIS, or PRODUCTION';
COMMENT ON COLUMN fraud.transactions.fraud_label_source IS 'How fraud label was determined: GROUND_TRUTH (IEEE-CIS), INJECTED (synthetic), UNKNOWN';
COMMENT ON COLUMN fraud.scoring_results.reason_codes IS 'JSON array of top contributing factors for the fraud score';
COMMENT ON COLUMN fraud.audit_log.triggered_by IS 'User or system that initiated the operation';
