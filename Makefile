.PHONY: run test clean

run:
	PYTHONPATH=src python -m retentioniq.pipeline run --customers 8000 --seed 42

test:
	PYTHONPATH=src python -m unittest discover -s tests

clean:
	rm -f data/raw/*.csv data/processed/*.csv artifacts/* reports/*
