#!/bin/sh
export PYTHONPATH=.
python3 -m experiments.main sens 2>&1 | grep -v Warn
python3 -m experiments.coverage_floor uci_har 100 6 2>&1 | grep -v Warn
python3 -m experiments.coverage_floor pamap2 100 6 2>&1 | grep -v Warn
echo ALLDONE
