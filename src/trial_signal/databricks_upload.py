"""Push local silver Parquet files into a Unity Catalog volume on Databricks Free Edition.

Then run databricks/setup.sql in the SQL editor (or via the SQL connector) to register
them as Delta tables that dbt reads from.

Docs: https://docs.databricks.com/aws/en/dev-tools/sdk-python
      https://docs.databricks.com/aws/en/volumes/
"""
from __future__ import annotations

import os
from pathlib import Path

from trial_signal.config import SILVER_DIR


def upload_silver(local_dir: Path = SILVER_DIR) -> list[str]:
    from databricks.sdk import WorkspaceClient  # reads DATABRICKS_HOST / DATABRICKS_TOKEN

    catalog = os.getenv("DATABRICKS_CATALOG", "workspace")
    schema = os.getenv("DATABRICKS_SCHEMA", "trial_signal")
    w = WorkspaceClient()
    uploaded = []
    for f in sorted(local_dir.glob("*.parquet")):
        target = f"/Volumes/{catalog}/{schema}/landing/silver/{f.name}"
        with f.open("rb") as fh:
            w.files.upload(target, fh, overwrite=True)
        uploaded.append(target)
    return uploaded


if __name__ == "__main__":
    for path in upload_silver():
        print("uploaded", path)
