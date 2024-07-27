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
    --seed 2025
}

#lcdrig
# ITERATIONS=(0 1 2)
NUM_AGENTS=(3 4 5)

#ENVS=("N43W080" "N45W123" "N47W124")
ENVS=("temp_data" )

#pip install --upgrade numpy scikit-image
#pip install numpy==1.26.4

#  $PYTHON $PY_FILE  --config "AK/experiments/configs/ak.yaml" --strategy "distributed" --num-agents 3 --version 0 --env-name "N45W123" --seed 3525
PY_FILE="main_lcd_rig.py"
$PYTHON $PY_FILE --config "AK/experiments/configs/ak.yaml" --strategy "distributed" --num-agents 3 --version 1 --env-name "N45W123" --seed 3525 --save-video

# cd "/home/airlab/PycharmProjects/LCD_RIG/outputs/3525/N45W123/distributed/"
# ./make_video.sh $(pwd)