#!/bin/bash
home="/home/local/USHERBROOKE/rabj2301/Projects"

code="$home/sources/epi_ml/src/python/compute_shaps.py"
env="$home/epilap/venv-epilap-pytorch/bin/activate"

. $env

input="$home/epilap/input"
hdf5_list="$input/hdf5_list/hg38_2022-epiatlas/estimator-debug-biotype-n400.list"
metadata="$input//metadata/merge_EpiAtlas_allmetadata-v11-mod.json"
chroms="$input/chromsizes/hg38.noy.chrom.sizes"

output="$home/epilap/output"
logdir="$output/logs/hg38_2022-epiatlas/shap"
model_dir="$output/models/split0"

# compute_shaps.py category hdf5 chromsize metadata logdir model
python $code "assay" $hdf5_list $chroms $metadata $logdir $model_dir
