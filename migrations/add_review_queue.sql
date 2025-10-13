-- Migration: Add Review Queue Table
-- T062: Queue low-confidence insights for manual review
-- Created: 2025-10-13

CREATE TABLE IF NOT EXISTS review_queue (
    id VARCHAR PRIMARY KEY,

    -- Insight content
    content TEXT NOT NULL,
    section VARCHAR NOT NULL,  -- 'helpful' or 'harmful'
    confidence REAL NOT NULL,
    rationale TEXT,

    -- Source context
    source_task_id VARCHAR NOT NULL,
    domain_id VARCHAR NOT NULL,

    -- Review status
    status VARCHAR NOT NULL DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    reviewer_id VARCHAR,

    -- Review decision
    review_notes TEXT,
    promoted_bullet_id VARCHAR
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_review_status_created ON review_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_review_domain_status ON review_queue(domain_id, status);
CREATE INDEX IF NOT EXISTS idx_review_source_task ON review_queue(source_task_id);

-- Comments
COMMENT ON TABLE review_queue IS 'Review queue for low-confidence insights requiring human approval';
COMMENT ON COLUMN review_queue.confidence IS 'Confidence score - insights <0.6 are queued';
COMMENT ON COLUMN review_queue.status IS 'Review status: pending, approved, rejected';
COMMENT ON COLUMN review_queue.promoted_bullet_id IS 'ID of bullet created if approved';
