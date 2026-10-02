#!/bin/sh
export PYTHONPATH=.
while ! grep -q ALLDONE results/rest.log; do sleep 30; done
python3 -m experiments.main extra 2>&1 | grep -v Warn
echo EXTRADONE
