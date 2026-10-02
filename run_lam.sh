#!/bin/sh
export PYTHONPATH=.
while pgrep -f "experiments.main ovh" > /dev/null; do sleep 20; done
python3 -m experiments.main lam 2>&1 | grep -v Warn
echo LAMDONE
