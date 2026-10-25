# Databricks notebook source
# Phase 7: chunks table -> AI Search (vector search) endpoint -> Delta Sync index.
# Import this file into your workspace (Workspace > Import) and run cell by cell.
# Docs: https://docs.databricks.com/aws/en/vector-search/create-vector-search
#
# Free Edition: ONE search endpoint, one search unit. Reuse the endpoint for every index.

# COMMAND ----------
# MAGIC %pip install --quiet databricks-ai-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
CATALOG = "workspace"
SCHEMA = "trial_signal"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.gold_trial_chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.trial_chunks_index"
ENDPOINT = "trial-signal-search"
# Check Serving > Endpoints for what your workspace actually offers before running.
EMBEDDING_ENDPOINT = "databricks-qwen3-embedding-0-6b"

# COMMAND ----------
# Delta Sync needs Change Data Feed on the source table.
spark.sql(f"""
CREATE OR REPLACE TABLE {SOURCE_TABLE}
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS SELECT * FROM read_files('/Volumes/{CATALOG}/{SCHEMA}/landing/silver/trial_chunks.parquet')
""")
display(spark.sql(f"SELECT section, COUNT(*) AS n FROM {SOURCE_TABLE} GROUP BY section"))

# COMMAND ----------
from databricks.ai_search.client import AISearchClient

client = AISearchClient()
try:
    client.get_endpoint(ENDPOINT)
except Exception:  # not found -> create it (Free Edition allows exactly one)
    client.create_endpoint(name=ENDPOINT, endpoint_type="STANDARD")
print(client.get_endpoint(ENDPOINT))  # wait until state is ONLINE before the next cell

# COMMAND ----------
index = client.create_delta_sync_index(
    endpoint_name=ENDPOINT,
    source_table_name=SOURCE_TABLE,
    index_name=INDEX_NAME,
    pipeline_type="TRIGGERED",  # sync on demand: cheaper than CONTINUOUS and fine for batch data
    primary_key="chunk_id",
    embedding_source_column="text",
    embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    columns_to_sync=["chunk_id", "nct_id", "drugs", "phase", "section", "criteria_type", "text"],
)

# COMMAND ----------
import time

index = client.get_index(index_name=INDEX_NAME)
while not index.describe().get("status", {}).get("detailed_state", "").startswith("ONLINE"):
    print("waiting for index...")
    time.sleep(15)
print("index ONLINE")

# COMMAND ----------
# Smoke test: one semantic query, one exact-term query, one filtered query.
for q, f in [
    ("patients with untreated brain metastases allowed", None),
    ("T-DM1 prior therapy", None),
    ("diarrhea prophylaxis loperamide", {"drugs LIKE": "neratinib"}),
]:
    kw = dict(query_text=q, columns=["chunk_id", "section", "text"], num_results=3, query_type="hybrid")
    if f:
        kw["filters"] = f
    res = index.similarity_search(**kw)
    print("\n##", q)
    for row in res["result"]["data_array"]:
        print(round(row[-1], 3), row[0], "|", row[2][:120].replace("\n", " "))

# COMMAND ----------
# After re-running `cli chunks` + upload + the CREATE TABLE cell, refresh the index:
# index.sync()
