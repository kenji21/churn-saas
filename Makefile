.PHONY: bronze silver reg-log-baseline random-forest-baseline xgboost-baseline dirty-cleanup

bronze:
	uv run python -m churn.cli ingest-bronze

silver:
	uv run python -m churn.cli build-silver

reg-log-baseline:
	uv run python -m churn.cli reg-log-baseline

random-forest-baseline:
	uv run python -m churn.cli random-forest-baseline

xgboost-baseline:
	uv run python -m churn.cli xgboost-baseline

dirty-cleanup:
	uv run python -m churn.cli dirty-cleanup
