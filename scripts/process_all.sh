#!/bin/bash
# Script global pour automatiser toutes les valeurs de N trouvées dans un dossier de scène

MODEL="$1"
SCENE="$2"

if [ -z "$MODEL" ] || [ -z "$SCENE" ]; then
  echo "Usage: $0 <MODEL> <SCENE>"
  exit 1
fi

# Récupération automatique des sous-dossiers de la forme scene01_N*
SCENE_DIR="data/images/${SCENE}"
N_DIRS=($(ls -d ${SCENE_DIR}/${SCENE}_N* 2>/dev/null))

if [ ${#N_DIRS[@]} -eq 0 ]; then
  echo "Aucun dossier trouvé dans $SCENE_DIR du type ${SCENE}_N*"
  exit 1
fi

for DIR in "${N_DIRS[@]}"; do
  N=$(find "$DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)

  echo "======== Traitement: $MODEL | $SCENE | N=$N ========"

  # 1. Inférence
  bash scripts/run_inference.sh "$MODEL" "$SCENE" "$N"

  # 2. Alignement ICP
  RECON="data/reconstructions/${MODEL}/${SCENE}_N${N}.ply"
  GT="data/gt_pointclouds/${SCENE}.ply"
  ALIGNED="data/reconstructions/${MODEL}/${SCENE}_N${N}_aligned.ply"
  TRANSFORM="results/metrics/${MODEL}/${SCENE}_N${N}_transform.npy"

  python scripts/align_icp.py \
    --source "$RECON" \
    --target "$GT" \
    --output "$ALIGNED" \
    --save_transform "$TRANSFORM"

  # Récupération du temps d'inférence
  TIME_FILE="results/metrics/${MODEL}/${SCENE}_N${N}_time.txt"
  INFER_TIME=$(cat "$TIME_FILE")

  # 3. Calcul des métriques
  python scripts/compute_metrics.py \
    --aligned "$ALIGNED" \
    --reference "$GT" \
    --model "$MODEL" \
    --scene "$SCENE" \
    --N "$N" \
    --inference_time "$INFER_TIME"
done
