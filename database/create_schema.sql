--- BEGIN create_schema.sql ---

-- ============================================================================
-- EMAIL CATEGORIZATION SYSTEM - DATABASE SCHEMA (NEON CLOUD VERSION)
-- Version: 1.0
-- Description: Production-grade schema with proper indexes and constraints
-- ============================================================================

-- Enable UUID extension for request IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE 1: emails
-- Purpose: Store raw email text inputs
-- ============================================================================

CREATE TABLE emails (
    id SERIAL PRIMARY KEY,
    email_text TEXT NOT NULL CHECK (length(email_text) > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Index for time-based queries
CREATE INDEX idx_emails_created_at ON emails(created_at DESC);

COMMENT ON TABLE emails IS 'Stores raw email text submitted for classification';
COMMENT ON COLUMN emails.email_text IS 'Raw unprocessed email content';

-- ============================================================================
-- TABLE 2: model_versions
-- Purpose: Track ML model versions and their performance metrics
-- ============================================================================

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    accuracy FLOAT CHECK (accuracy >= 0 AND accuracy <= 1),
    precision_score FLOAT CHECK (precision_score >= 0 AND precision_score <= 1),
    recall_score FLOAT CHECK (recall_score >= 0 AND recall_score <= 1),
    f1_score FLOAT CHECK (f1_score >= 0 AND f1_score <= 1),
    is_active BOOLEAN DEFAULT FALSE,
    model_path TEXT,
    vectorizer_path TEXT,
    training_samples INTEGER,
    training_metrics JSONB,
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Index for quick lookup of active model
CREATE INDEX idx_model_versions_active ON model_versions(is_active) WHERE is_active = true;
CREATE INDEX idx_model_versions_version ON model_versions(version);

COMMENT ON TABLE model_versions IS 'Registry of all trained model versions with metrics';
COMMENT ON COLUMN model_versions.is_active IS 'Only one model should be active at a time';
COMMENT ON COLUMN model_versions.training_metrics IS 'JSON object containing confusion matrix, per-class metrics, etc.';

-- ============================================================================
-- TABLE 3: predictions
-- Purpose: Log all model predictions with confidence and version tracking
-- ============================================================================

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    email_id INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    predicted_label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    prediction_probabilities JSONB,
    processing_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Indexes for performance monitoring and queries
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_label ON predictions(predicted_label);
CREATE INDEX idx_predictions_confidence ON predictions(confidence);
CREATE INDEX idx_predictions_email_id ON predictions(email_id);
CREATE INDEX idx_predictions_model_version_id ON predictions(model_version_id);

COMMENT ON TABLE predictions IS 'Logs every prediction made by the system';
COMMENT ON COLUMN predictions.prediction_probabilities IS 'JSON object with probabilities for all classes';
COMMENT ON COLUMN predictions.processing_time_ms IS 'Time taken for preprocessing + inference in milliseconds';

-- ============================================================================
-- TABLE 4: feedback
-- Purpose: Store user corrections for continuous learning
-- ============================================================================

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER UNIQUE NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    corrected_label VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    feedback_source VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Index for tracking correction patterns
CREATE INDEX idx_feedback_created_at ON feedback(created_at DESC);
CREATE INDEX idx_feedback_corrected_label ON feedback(corrected_label);

COMMENT ON TABLE feedback IS 'User corrections for wrong predictions - drives retraining';
COMMENT ON COLUMN feedback.prediction_id IS 'One-to-one relationship with predictions';
COMMENT ON COLUMN feedback.feedback_source IS 'E.g., manual, automated, imported';

-- ============================================================================
-- TABLE 5: audit_logs
-- Purpose: Track all API requests for debugging and monitoring
-- ============================================================================

CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    request_id UUID DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    latency_ms FLOAT,
    user_id VARCHAR(100),
    ip_address INET,
    error_message TEXT,
    request_payload JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Index for time-series analysis
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_logs_endpoint ON audit_logs(endpoint);
CREATE INDEX idx_audit_logs_status_code ON audit_logs(status_code);

COMMENT ON TABLE audit_logs IS 'Complete audit trail of all API requests';

-- ============================================================================
-- TABLE 6: inference_metadata
-- Purpose: Track aggregate statistics for monitoring dashboards
-- ============================================================================

CREATE TABLE inference_metadata (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    category_distribution JSONB,
    avg_confidence FLOAT,
    low_confidence_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX idx_inference_metadata_date ON inference_metadata(date);

COMMENT ON TABLE inference_metadata IS 'Daily aggregated statistics for monitoring';

-- ============================================================================
-- TABLE 7: retraining_jobs
-- Purpose: Track retraining pipeline executions
-- ============================================================================

CREATE TABLE retraining_jobs (
    id SERIAL PRIMARY KEY,
    trigger_reason VARCHAR(200),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(50) DEFAULT 'running',
    new_model_version VARCHAR(50),
    training_samples INTEGER,
    validation_accuracy FLOAT,
    promoted BOOLEAN DEFAULT FALSE,
    error_log TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_retraining_jobs_status ON retraining_jobs(status);
CREATE INDEX idx_retraining_jobs_created_at ON retraining_jobs(created_at DESC);

COMMENT ON TABLE retraining_jobs IS 'Tracks automated and manual retraining executions';

-- ============================================================================
-- VIEWS FOR MONITORING
-- ============================================================================

-- View: Recent predictions with feedback status
CREATE VIEW v_recent_predictions AS
SELECT 
    p.id,
    p.created_at,
    e.email_text,
    p.predicted_label,
    p.confidence,
    mv.version AS model_version,
    f.corrected_label,
    CASE WHEN f.id IS NOT NULL THEN 'corrected' ELSE 'not_corrected' END AS feedback_status
FROM predictions p
JOIN emails e ON p.email_id = e.id
JOIN model_versions mv ON p.model_version_id = mv.id
LEFT JOIN feedback f ON p.id = f.prediction_id
ORDER BY p.created_at DESC;

-- View: Model performance metrics
CREATE VIEW v_model_performance AS
SELECT 
    mv.version,
    mv.accuracy,
    mv.f1_score,
    mv.is_active,
    COUNT(p.id) AS total_predictions,
    COUNT(f.id) AS total_corrections,
    CASE 
        WHEN COUNT(p.id) > 0 THEN ROUND((COUNT(f.id)::DECIMAL / COUNT(p.id) * 100), 2)
        ELSE 0 
    END AS correction_rate_percent
FROM model_versions mv
LEFT JOIN predictions p ON mv.id = p.model_version_id
LEFT JOIN feedback f ON p.id = f.prediction_id
GROUP BY mv.id, mv.version, mv.accuracy, mv.f1_score, mv.is_active
ORDER BY mv.created_at DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Ensure only one active model at a time
CREATE OR REPLACE FUNCTION ensure_single_active_model()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active = TRUE THEN
        UPDATE model_versions SET is_active = FALSE WHERE id != NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Enforce single active model
CREATE TRIGGER trigger_single_active_model
BEFORE INSERT OR UPDATE ON model_versions
FOR EACH ROW
WHEN (NEW.is_active = TRUE)
EXECUTE FUNCTION ensure_single_active_model();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert a placeholder model version (will be replaced by actual training)
INSERT INTO model_versions (
    version, 
    accuracy, 
    precision_score, 
    recall_score, 
    f1_score, 
    is_active,
    training_samples
) VALUES (
    'v0.0.0-placeholder',
    0.0,
    0.0,
    0.0,
    0.0,
    FALSE,
    0
);
