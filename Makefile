.PHONY: bronze silver dirty-cleanup

bronze:
	uv run python -m churn.cli ingest-bronze

silver:
	uv run python -m churn.cli build-silver

dirty-cleanup:
	uv run python -m churn.cli dirty-cleanup
