# Reproduce every number in the paper. `make all` is the whole contract.
PY ?= python
DATASET ?=                    # set to a local snapshot to score offline
REF = $(if $(DATASET),--references file --dataset $(DATASET),)

.PHONY: all asr diacritization g2p a2p examples iaa audio lint clean

all: asr diacritization g2p a2p examples iaa

asr:
	$(PY) -m shams_eval.cli score asr $(REF) --by-source

diacritization:
	$(PY) -m shams_eval.cli score diacritization $(REF) --by-source

g2p:
	$(PY) -m shams_eval.cli score g2p $(REF) --by-source

a2p:
	$(PY) -m shams_eval.cli score a2p $(REF) --by-source

examples:
	$(PY) -m shams_eval.cli examples $(REF)

iaa:
	$(PY) -m shams_eval.cli iaa

# Not part of `all`: this one reaches the network, needs ffmpeg, and needs the
# Casablanca terms accepted. AUDIO overrides where the segments are written.
AUDIO ?= audio
audio:
	$(PY) -m shams_eval.cli build-audio --out $(AUDIO) $(REF)
	$(PY) -m shams_eval.cli verify-audio $(AUDIO)

lint:
	ruff check shams_eval

clean:
	rm -rf results/*.json results/*.md results/tables
