.PHONY: test demo cli-check audit

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 examples/full_engine_demo.py

.PHONY: demo-light

demo-light:
	PYTHONPATH=src python3 examples/run_demo.py --show-prompt 0

cli-check:
	PYTHONPATH=src python3 -m memcatalog --help

audit:
	! rg -n -i '([v][0-9]+|memcat_[v]|/[U]sers/|/[h]ome/|/[m]nt/)' . --hidden --glob '!.git/**' --glob '!results/**' --glob '!Makefile'
