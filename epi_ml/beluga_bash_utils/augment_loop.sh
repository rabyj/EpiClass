#!/bin/bash
# ---------------------------
# Run augment_predict_file.py for all separate output splits, and then merged.
# I.e. concatenate results vertically. (not horizontally like "merge_all_predictions.py")
# Use when predictions have overlapping md5sum.
# ---------------------------

# -- Paths setup --
gen_path="/home/local/USHERBROOKE/rabj2301/Projects"
source "${gen_path}/epilap/venv-epilap-pytorch/bin/activate"

input_path="${gen_path}/epilap/input"
output_path="${gen_path}/epilap/output/logs/hg38_2022-epiatlas/predict_new"

code="${gen_path}/sources/epi_ml/epi_ml/python/utils"

# -- Actual code --
metadata="${input_path}/metadata/merge_EpiAtlas_allmetadata-v11-mod.json"

for i in {0..9}; do
  to_augment="${output_path}/split${i}*.csv"
  python ${code}/augment_predict_file.py --correct-true assay --all-categories ${to_augment} ${metadata}
done

final_output="full_test_prediction_100kb_all_none_predict-252_augmented_all.csv"
cat ${output_path}/split*augmented*.csv | sort -ru > ${final_output}
