setup:
	python -m venv .venv
	./.venv/bin/python -m pip install -r requirements.txt

run:
	./.venv/bin/python main.py

smoke:
	./.venv/bin/python main.py --step smoke_test

analysis:
	./.venv/bin/python main.py --step analysis

preflight:
	./.venv/bin/python main.py --step preflight

clean-results:
	rm -rf data/processed/* data/results/* reports/figures/* reports/*.md
