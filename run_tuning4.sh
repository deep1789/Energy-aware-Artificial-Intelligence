#!/bin/sh
export PYTHONPATH=.
for ds in uci_har pamap2 speech; do
  for phi in 0.1 0.25; do
    [ -f results/best4_${ds}_${phi}.json ] || python3 -m experiments.tune4 $ds $phi 80 2>&1 | grep -v Warn
  done
done
echo TUNEDONE
