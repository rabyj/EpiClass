#!/bin/bash
gen_path="/home/rabyj/project-rabyj"
input_path="${gen_path}/epilap/input"
output_path="${gen_path}/epilap/output/logs"

# use correct environment
source /lustre07/scratch/rabyj/envs/epilap/bin/activate

# --- choose category + source files ---
category="cell_type" # IMPORTANT
resolution="100kb" # IMPORTANT

release="2022-epiatlas"
assembly="hg38"

basename="${resolution}_all_none"
list_name="${basename}_plus" # IMPORTANT

dataset=${assembly}"_"${release} # ex: hg38_2018-10

export LAYER_SIZE="3000" # IMPORTANT
export NB_LAYER="1"

log="${output_path}/${release}/${assembly}_${basename}_pearson/${category}_${NB_LAYER}l_${LAYER_SIZE}n" # IMPORTANT# IMPORTANT# IMPORTANT# IMPORTANT
# log="${log}/10fold-l2" # IMPORTANT
log="${log}/10fold"

export LOG="${log}"
export NO_TRUE="False"

set -e
program_path="${gen_path}/sources/epi_ml/epi_ml"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py --exists ${log}"
python ${program_path}/python/utils/check_dir.py --exists ${log}

cd ${log}
printf '\n%s\n' "Launching following command"
printf '%s\n' "cat split*/validation_prediction.csv | sort -ru > full-10fold-validation_prediction.csv"
cat split*/validation_prediction.csv | sort -ru > full-10fold-validation_prediction.csv

# categories="data_generating_centre epirr_id uuid track_type"
to_augment="${log}/full-10fold-validation_prediction.csv"
# to_augment="${log}/validation_prediction.csv"
metadata="${input_path}/metadata/merged_EpiAtlas_newCT.json"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories
