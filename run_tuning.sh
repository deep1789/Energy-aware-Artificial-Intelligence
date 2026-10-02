#!/bin/sh
# Resumable: finished jobs are cached under results/raw
export PYTHONPATH=.
for ds in uci_har pamap2; do
  for phi in 0.1 0.25; do
    [ -f results/best3_${ds}_${phi}.json ] || python3 -m experiments.tune2 $ds $phi 80 2>&1 | grep -v Warn
  done
done
