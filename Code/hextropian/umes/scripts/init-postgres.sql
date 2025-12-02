-- UMES Database Initialization Script
-- Creates keycloak database for Keycloak service
-- Enables required PostgreSQL extensions

-- Create keycloak database for Keycloak
CREATE DATABASE keycloak;

-- Connect to umes database
\c umes;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable Row-Level Security (will be configured per table in migrations)
-- This is a placeholder - actual RLS policies will be created by Alembic migrations

COMMENT ON DATABASE umes IS 'UMES - Unified Management of Entitlements and Identity Subsystem';
