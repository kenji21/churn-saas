.PHONY: bronze silver reg-log-baseline random-forest-baseline random-forest-ordinal-baseline random-forest-tune xgboost-baseline xgboost-tune mlflow-ui api dirty-cleanup

MLFLOW_HOST := 127.0.0.1
MLFLOW_PORT := 5000
API_PORT := 8000

bronze:
	uv run python -m churn.cli ingest-bronze

silver:
	uv run python -m churn.cli build-silver

reg-log-baseline:
	uv run python -m churn.cli reg-log-baseline

random-forest-baseline:
	uv run python -m churn.cli random-forest-baseline

random-forest-ordinal-baseline:
	uv run python -m churn.cli random-forest-ordinal-baseline

random-forest-tune:
	uv run python -m churn.cli random-forest-tune

xgboost-baseline:
	uv run python -m churn.cli xgboost-baseline

xgboost-tune:
	uv run python -m churn.cli xgboost-tune

mlflow-ui:
	@if [ "$$(uname)" = "Darwin" ]; then (sleep 1 && open http://$(MLFLOW_HOST):$(MLFLOW_PORT)) & fi
	uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

api:
	@if [ "$$(uname)" = "Darwin" ]; then (sleep 2 && open http://$(MLFLOW_HOST):$(API_PORT)/docs) & fi
	uv run uvicorn churn.api.app:app --host $(MLFLOW_HOST) --port $(API_PORT)

dirty-cleanup:
	uv run python -m churn.cli dirty-cleanup
