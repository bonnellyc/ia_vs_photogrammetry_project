#!/bin/bash
# Script pour lancer inférence par modèle et par scène 
# avec mesure du temps d'inference

MODEL=$1           
SCENE=$2           
N_IMAGES=$3        

INPUT_DIR="data/images/${SCENE}/${SCENE}_N${N_IMAGES}/"
OUTPUT_DIR="data/reconstructions/${MODEL}/"
OUTPUT_FILE="${OUTPUT_DIR}/${SCENE}_N${N_IMAGES}.ply"
TIME_FILE="results/metrics/${MODEL}/${SCENE}_N${N_IMAGES}_time.txt"

mkdir -p "$OUTPUT_DIR"
mkdir -p "results/metrics/${MODEL}"

echo ">> Inférence $MODEL | Scène: $SCENE | N: $N_IMAGES"

START=$(date +%s.%N)

docker-compose run --rm ${MODEL,,}_runner python scripts/model_wrappers/${MODEL,,}_runner.py \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT_FILE" \
  --N "$N_IMAGES"

END=$(date +%s.%N)
INFERENCE_TIME=$(echo "$END - $START" | bc)

echo "$INFERENCE_TIME" > "$TIME_FILE"
echo "Temps d'inférence : $INFERENCE_TIME secondes"