#!/bin/sh
export PYTHONPATH=.
while ! grep -q TUNEDONE results/tuning4.log; do sleep 60; done
for ds in uci_har pamap2 speech; do cp results/best4_${ds}_0.25.json results/best4_${ds}_1.0.json; done
python3 -m experiments.main main 2>&1 | grep -v Warn
python3 -m experiments.main sens 2>&1 | grep -v Warn
python3 -m experiments.main ovh 2>&1 | grep -v Warn
python3 -m experiments.main lam 2>&1 | grep -v Warn
echo ALL3DONE
