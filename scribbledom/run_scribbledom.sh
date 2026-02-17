#!/bin/bash
set -e

CONFIG_FILE=$1

JOB_DIR=$(readlink -f "$(dirname "$(dirname "$CONFIG_FILE")")")



SPACE_RANGER_FOLDER=$(jq -r '.space_ranger_output_directory // "raw_gene_x"' "$CONFIG_FILE")
PREPROCESSED_DATA_FOLDER=$(jq -r '.preprocessed_data_folder // "preprocessed_data"' "$CONFIG_FILE")
MATRIX_REPRESENTATION_FOLDER=$(jq -r '.matrix_represenation_of_ST_data_folder // "matrix_representation"' "$CONFIG_FILE")
MODEL_OUTPUTS_FOLDER=$(jq -r '.model_output_folder // "model_outputs"' "$CONFIG_FILE")
FINAL_OUTPUTS_FOLDER=$(jq -r '.final_output_folder // "final_outputs"' "$CONFIG_FILE")
DATASET=$(jq -r '.dataset' "$CONFIG_FILE")
SAMPLE=$(jq -r '.samples[0]' "$CONFIG_FILE")



mkdir -p "$JOB_DIR/outputs"
mkdir -p "$JOB_DIR/outputs/$PREPROCESSED_DATA_FOLDER"
mkdir -p "$JOB_DIR/outputs/$MATRIX_REPRESENTATION_FOLDER"
mkdir -p "$JOB_DIR/outputs/$MODEL_OUTPUTS_FOLDER"
mkdir -p "$JOB_DIR/outputs/$FINAL_OUTPUTS_FOLDER"


# -----------------------------
# Redirect tool outputs
# -----------------------------
ln -sfn "$JOB_DIR/outputs/$MODEL_OUTPUTS_FOLDER" "/opt/ScribbleDom/$MODEL_OUTPUTS_FOLDER"
ln -sfn "$JOB_DIR/outputs/$FINAL_OUTPUTS_FOLDER" "/opt/ScribbleDom/$FINAL_OUTPUTS_FOLDER"
ln -sfn "$JOB_DIR/outputs/$PREPROCESSED_DATA_FOLDER" "/opt/ScribbleDom/$PREPROCESSED_DATA_FOLDER"
ln -sfn "$JOB_DIR/outputs/$MATRIX_REPRESENTATION_FOLDER" "/opt/ScribbleDom/$MATRIX_REPRESENTATION_FOLDER"

ln -sfn "$JOB_DIR/$SPACE_RANGER_FOLDER" "/opt/ScribbleDom/$SPACE_RANGER_FOLDER"

cd /opt/ScribbleDom

# -----------------------------
# Run pipeline
# -----------------------------
Rscript get_genex_data_from_10x_h5.R ${CONFIG_FILE}


# -----------------------------
# Materialize staged inputs
# -----------------------------
SCHEMA=$(jq -r '.schema' "$CONFIG_FILE")

if [ "$SCHEMA" = "expert" ]; then
  SCRIBBLE_SRC="$JOB_DIR/staged_inputs/manual_scribble.csv"
  SCRIBBLE_DST="$JOB_DIR/outputs/$PREPROCESSED_DATA_FOLDER/$DATASET/$SAMPLE"

  mkdir -p "$SCRIBBLE_DST"
  cp "$SCRIBBLE_SRC" "$SCRIBBLE_DST/manual_scribble.csv"
fi


python visium_data_to_matrix_representation_converter.py --params "$CONFIG_FILE"
python scribble_dom.py --params "$CONFIG_FILE"
python best_model_estimator.py --params "$CONFIG_FILE"
python show_results.py --params "$CONFIG_FILE"

echo "✅ Done. Outputs in $JOB_DIR/outputs"
