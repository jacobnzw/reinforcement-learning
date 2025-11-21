#!/usr/bin/bash
BACKEND_STORE_URI="sqlite:///mlflow_tracking/mlruns.db"
ARTIFACTS_DESTINATION="./mlflow_tracking/mlartifacts"

uv run mlflow server \
--backend-store-uri $BACKEND_STORE_URI \
--artifacts-destination $ARTIFACTS_DESTINATION \
--default-artifact-root $ARTIFACTS_DESTINATION
