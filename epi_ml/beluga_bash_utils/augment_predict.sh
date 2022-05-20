#!/bin/bash

project_path="/lustre03/project/6007017/rabyj/epi_ml_project"

code="${project_path}/epi_ml/epi_ml/python/augment_predict_file.py"
logs_dir="${project_path}/sub/logs"
metadata_dir="${project_path}/metadata"

predict_file="${logs_dir}/2018-10/hg19_100kb_all_none_pearson/major_cell_type_alt_1l_3000n/validation_predict.csv"
metatada_file="${metadata_dir}/hg19_2018-10_final.json"

. ${project_path}/epi_ml/venv/bin/activate

python $code $predict_file $metatada_file assay
