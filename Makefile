# retrieval-bridge — convenience targets.
# Assumes an activated virtualenv (python -m venv .venv && source .venv/bin/activate),
# or override PY, e.g.:  make demo PY=.venv/bin/python

PY ?= python

.PHONY: help install corpus seed demo bench test all clean

help:
	@echo "Targets:"
	@echo "  install   pip install -e . (base: local embeddings + LanceDB, zero keys)"
	@echo "  corpus    generate the synthetic ticket corpus"
	@echo "  seed      embed + index the corpus (LanceDB) and structured store (SQLite)"
	@echo "  demo      run the end-to-end PromptQL-style query plan"
	@echo "  bench     hybrid vs pure-vector on exact-token lookups"
	@echo "  test      run the test suite"
	@echo "  all       corpus + seed + demo"
	@echo "  clean     remove local data stores"
	@echo ""
	@echo "Backend/embedder via env, e.g.:  RETRIEVAL_BRIDGE_BACKEND=turbopuffer make seed"

install:
	$(PY) -m pip install -e .

corpus:
	$(PY) data/generate_corpus.py

seed:
	$(PY) scripts/seed.py

demo:
	$(PY) scripts/demo.py

bench:
	$(PY) scripts/bench.py

test:
	$(PY) -m pytest -q

all: corpus seed demo

clean:
	rm -rf .lancedb .structured.db
	@echo "Removed local data stores (regenerate with: make seed)"
