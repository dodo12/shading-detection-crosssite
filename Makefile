PY ?= python3
.PHONY: all prepare analysis figures tables check clean
all: prepare analysis figures tables check
prepare:
	$(PY) src/prepare_reunion.py
analysis:
	$(PY) src/feasibility_localisation.py
	$(PY) src/detect_transfer.py
	$(PY) src/extra_analyses.py
figures:
	$(PY) src/make_figures.py
tables:
	$(PY) src/make_supplementary_tables.py
check:
	$(PY) tests/check_key_numbers.py
clean:
	rm -f results/*.json results/tables/*.csv data/derived/reunion_hourly_qc.csv figures/*.png figures/*.pdf docs/SUPPLEMENTARY_TABLES.md
