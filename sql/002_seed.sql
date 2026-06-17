-- ============================================================================
-- Fraud Detection Pipeline - Seed Data
-- ============================================================================
-- Initial seed data for development and testing
-- For production, use the synthetic data generator or IEEE-CIS loader
-- ============================================================================

SET search_path TO fraud, public;

-- ============================================================================
-- SAMPLE MERCHANTS (for reference data)
-- ============================================================================

-- Insert some reference merchants for testing
INSERT INTO fraud.transactions (
    transaction_id, timestamp, amount, currency, merchant_id, merchant_category,
    card_id, card_type, card_country, customer_id, device_id, ip_address,
    email_domain, billing_country, shipping_country, is_online, product_category,
    data_source, is_fraud, fraud_label_source
) VALUES
    -- Normal transactions
    ('seed_txn_001', NOW() - INTERVAL '1 hour', 42.50, 'USD', 'merchant_walmart_001', 'grocery',
     'card_visa_001', 'VISA', 'USA', 'cust_001', 'device_001', '192.168.1.100',
     'gmail.com', 'USA', 'USA', TRUE, 'groceries',
     'SYNTHETIC', FALSE, 'INJECTED'),

    ('seed_txn_002', NOW() - INTERVAL '2 hours', 156.99, 'USD', 'merchant_amazon_001', 'electronics',
     'card_visa_001', 'VISA', 'USA', 'cust_001', 'device_001', '192.168.1.100',
     'gmail.com', 'USA', 'USA', TRUE, 'electronics',
     'SYNTHETIC', FALSE, 'INJECTED'),

    ('seed_txn_003', NOW() - INTERVAL '3 hours', 8.50, 'USD', 'merchant_starbucks_001', 'food_beverage',
     'card_mc_001', 'MASTERCARD', 'USA', 'cust_002', 'device_002', '10.0.0.50',
     'yahoo.com', 'USA', 'USA', FALSE, 'coffee',
     'SYNTHETIC', FALSE, 'INJECTED'),

    -- Suspicious transactions (for testing fraud detection)
    ('seed_txn_004', NOW() - INTERVAL '30 minutes', 9999.99, 'USD', 'merchant_jewelry_001', 'jewelry',
     'card_visa_002', 'VISA', 'USA', 'cust_003', 'device_003', '185.220.101.1',
     'tempmail.com', 'USA', 'NGA', TRUE, 'luxury',
     'SYNTHETIC', TRUE, 'INJECTED'),

    ('seed_txn_005', NOW() - INTERVAL '29 minutes', 8500.00, 'USD', 'merchant_electronics_001', 'electronics',
     'card_visa_002', 'VISA', 'USA', 'cust_003', 'device_003', '185.220.101.1',
     'tempmail.com', 'USA', 'NGA', TRUE, 'electronics',
     'SYNTHETIC', TRUE, 'INJECTED'),

    -- Velocity anomaly (same card, multiple transactions in short time)
    ('seed_txn_006', NOW() - INTERVAL '10 minutes', 500.00, 'USD', 'merchant_gas_001', 'gas_station',
     'card_amex_001', 'AMEX', 'USA', 'cust_004', 'device_004', '72.45.100.200',
     'outlook.com', 'USA', 'USA', FALSE, 'fuel',
     'SYNTHETIC', TRUE, 'INJECTED'),

    ('seed_txn_007', NOW() - INTERVAL '9 minutes', 450.00, 'USD', 'merchant_gas_002', 'gas_station',
     'card_amex_001', 'AMEX', 'USA', 'cust_004', 'device_005', '103.21.50.75',
     'outlook.com', 'USA', 'USA', FALSE, 'fuel',
     'SYNTHETIC', TRUE, 'INJECTED'),

    ('seed_txn_008', NOW() - INTERVAL '8 minutes', 475.00, 'USD', 'merchant_gas_003', 'gas_station',
     'card_amex_001', 'AMEX', 'USA', 'cust_004', 'device_006', '45.33.77.129',
     'outlook.com', 'CAN', 'CAN', FALSE, 'fuel',
     'SYNTHETIC', TRUE, 'INJECTED')

ON CONFLICT (transaction_id) DO NOTHING;

-- ============================================================================
-- SAMPLE AUDIT LOG ENTRIES
-- ============================================================================

INSERT INTO fraud.audit_log (
    batch_id, operation, status, records_processed, records_flagged,
    duration_seconds, triggered_by, source_system, metadata
) VALUES
    ('seed_batch_001', 'EXTRACT', 'COMPLETED', 8, NULL, 0.125, 'seed_script', 'SQL',
     '{"description": "Initial seed data load"}'),
    ('seed_batch_001', 'VALIDATE', 'COMPLETED', 8, NULL, 0.050, 'seed_script', 'SQL',
     '{"expectation_suite": "basic_validation", "success": true}'),
    ('seed_batch_001', 'SCORE', 'COMPLETED', 8, 5, 0.350, 'seed_script', 'SQL',
     '{"models": ["rules", "isolation_forest", "autoencoder"]}')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Show transaction counts by fraud status
SELECT
    is_fraud,
    COUNT(*) as count,
    AVG(amount) as avg_amount,
    MAX(amount) as max_amount
FROM fraud.transactions
WHERE data_source = 'SYNTHETIC'
GROUP BY is_fraud;

-- Show recent audit log
SELECT
    batch_id,
    operation,
    status,
    records_processed,
    records_flagged,
    duration_seconds
FROM fraud.audit_log
ORDER BY timestamp DESC
LIMIT 10;
