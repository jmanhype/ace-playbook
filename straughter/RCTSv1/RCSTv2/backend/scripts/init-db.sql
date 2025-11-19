-- Database initialization script for RCST v2
-- Enables required PostgreSQL extensions for the application
--
-- This script runs automatically on first database creation via docker-compose
-- or can be run manually: psql -U rcst_user -d rcst_db -f init-db.sql

-- Enable UUID generation (required for all primary keys)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgvector for semantic search (384D embeddings from sentence-transformers)
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extensions are enabled
SELECT * FROM pg_extension WHERE extname IN ('uuid-ossp', 'vector');

-- Create database-level configuration for Row-Level Security
-- Set default search path to prevent schema hijacking attacks
ALTER DATABASE rcst_db SET search_path TO public, pg_catalog;

-- Grant necessary permissions to application user
GRANT USAGE ON SCHEMA public TO rcst_user;
GRANT CREATE ON SCHEMA public TO rcst_user;

-- Log successful initialization
DO $$
BEGIN
  RAISE NOTICE 'RCST v2 database initialization complete';
  RAISE NOTICE 'Extensions enabled: uuid-ossp, pgvector';
  RAISE NOTICE 'Ready for Alembic migrations';
END $$;
