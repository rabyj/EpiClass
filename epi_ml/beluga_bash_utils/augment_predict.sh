#!/bin/bash
gen="/lustre06/project/6007017/rabyj"
project_path="${gen}/epilap"

. ${project_path}/venv-torch/bin/activate

metadata_dir="${project_path}/input/metadata"
metatada_file="${metadata_dir}/hg38_2022-epiatlas_harmonizedv8.json"

logs_dir="${project_path}/output/logs"
predict_file="${logs_dir}/2022-epiatlas/hg38_100kb_all_none_pearson/assay_1l_3000n/10fold/full-10fold-validation_prediction.csv"

categories="data_generating_centre epirr_id uuid track_type"

code="${gen}/sources/epi_ml/epi_ml/python/utils/augment_predict_file.py"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python $code $predict_file $metatada_file $categories"
python $code $predict_file $metatada_file $categories
