#!/usr/bin/env bash
PYTHON="/home/redwan/anaconda3/envs/rig/bin/python"

srig()
{
  PY_FILE="main_single_rig.py"
  $PYTHON $PY_FILE  --config "AK/experiments/configs/ak.yaml" \
    --env-name "N45W123" \
    --strategy "myopic" \
    --seed 0
}

srig