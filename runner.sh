#!/usr/bin/env bash
PYTHON="/home/redwan/anaconda3/envs/rig/bin/python"
PYTHON="/home/airlab/anaconda3/envs/LCD_RIG/bin/python"

srig()
{
  PY_FILE="main_single_rig.py"
  $PYTHON $PY_FILE  --config "AK/experiments/configs/ak.yaml" \
    --env-name "N45W123" \
    --strategy "myopic" \
    --seed 0
}

lcdrig()
{
  PY_FILE="main_lcd_rig.py"
  $PYTHON $PY_FILE  --config "AK/experiments/configs/ak.yaml" \
    --env-name "N45W123" \
    --strategy "distributed" \
    --num-agents 4 \
    --seed 1
}

lcdrig