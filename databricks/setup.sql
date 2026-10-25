-- Run once in the Databricks SQL editor (Free Edition, serverless SQL warehouse).
-- Medallion layout: landing volume (files) -> silver Delta tables -> gold built by dbt.

CREATE SCHEMA IF NOT EXISTS workspace.trial_signal;
CREATE VOLUME IF NOT EXISTS workspace.trial_signal.landing;

-- After `python -m trial_signal.databricks_upload`, register silver files as Delta tables.
CREATE OR REPLACE TABLE workspace.trial_signal.silver_trials AS
SELECT * FROM read_files('/Volumes/workspace/trial_signal/landing/silver/trials.parquet');

CREATE OR REPLACE TABLE workspace.trial_signal.silver_arm_adverse_events AS
SELECT * FROM read_files('/Volumes/workspace/trial_signal/landing/silver/arm_adverse_events.parquet');

CREATE OR REPLACE TABLE workspace.trial_signal.silver_rejects AS
SELECT * FROM read_files('/Volumes/workspace/trial_signal/landing/silver/rejects.parquet');

-- Phase 7: the chunks table is created with Change Data Feed in databricks/02_vector_index.py.

-- Phase 7: the RAG app runs as its own service principal. Grant it read-only access.
-- Find the principal's application ID on the app's page (Compute > Apps > your app > Authorization).
-- GRANT USE CATALOG ON CATALOG workspace TO `<app-service-principal-id>`;
-- GRANT USE SCHEMA, SELECT ON SCHEMA workspace.trial_signal TO `<app-service-principal-id>`;

-- Sanity checks
SELECT COUNT(*) AS trials FROM workspace.trial_signal.silver_trials;
SELECT seriousness, COUNT(*) FROM workspace.trial_signal.silver_arm_adverse_events GROUP BY 1;
