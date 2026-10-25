# Databricks Free Edition setup (~30 min)

1. Sign up for **Databricks Free Edition** (search "Databricks Free Edition signup"). It gives you a
   serverless SQL warehouse, Unity Catalog with a `workspace` catalog, and notebooks.
2. **Settings → Developer → Access tokens** → generate a personal access token.
   Put host + token in `.env` (see `.env.example`).
3. **SQL Warehouses → your warehouse → Connection details** → copy the HTTP path into `.env`.
4. Open the SQL editor, run the first two statements of `setup.sql` (schema + volume).
5. Locally: `python -m trial_signal.cli silver` then `python -m trial_signal.databricks_upload`.
6. Run the rest of `setup.sql` to register the silver Delta tables.
7. `cd dbt && dbt build` builds the gold layer on top.

Free Edition has compute and quota limits. This dataset is tiny (thousands of rows), so you
won't hit them, but stop the warehouse when you're done.

Docs to skim first:
- Medallion architecture: https://www.databricks.com/glossary/medallion-architecture
- Volumes: https://docs.databricks.com/aws/en/volumes/
- `read_files`: https://docs.databricks.com/aws/en/sql/language-manual/functions/read_files
- dbt + Databricks: https://docs.getdbt.com/docs/core/connect-data-platform/databricks-setup
