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
    --strategy "distributed" \
    --num-agents $1 \
    --version $2 \
    --env-name $3 \
    --seed 2024
}

#lcdrig
ITERATIONS=(0 1 2)
NUM_AGENTS=(3 4 5)

ENVS=("N43W080" "N45W123" "N47W124")

for j in ${ITERATIONS[@]}
do
  for i in ${NUM_AGENTS[@]}
  do
    for env in ${ENVS[@]}
    do
      echo "[+] exp $env running with version $j with num agents $i"
      lcdrig $i $j $env
    done
  done
done