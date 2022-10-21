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

cd ${output_path}
for i in {0..9}; do
  to_augment=$(find . -maxdepth 1 -name "*split${i}*" | grep -v "augmented")
  python ${code}/augment_predict_file.py ${to_augment} ${metadata} --correct-true assay --all-categories
  augmented="${output_path}/*split${i}*augmented-all.csv"

  #add filename as last column, with header "ID2"
  filename="$(basename -- ${augmented})"
  awk -v to_add=${filename} 'BEGIN{FS=OFS=","}{print $0 OFS to_add}' ${augmented} > tmp && mv tmp ${augmented}
  sed -i "0,/${filename}/{s//ID2/}" ${augmented}
done
wait

final_output="$(ls ${filename} | sed 's/split[0-9]//').merged" #single quotes important
cat ${output_path}/*split*augmented*.csv | sort -ru > ${final_output}
