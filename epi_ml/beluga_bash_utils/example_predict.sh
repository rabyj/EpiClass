#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-jacquesp
#SBATCH --job-name=2019-11_predict
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --mail-user=joanny.raby@usherbrooke.ca
#SBATCH --mail-type=END

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print =========================================="
fi

gen_path="/home/rabyj/project-rabyj"
input_path="${gen_path}/epilap/input"
output_path="${gen_path}/epilap/output/logs"

# use correct environment
source ${gen_path}/epilap/venv-torch/bin/activate

# --- choose category + hparams + source files ---
category="assay"
hparams="human_base.json"
release="2019-11"
assembly="hg38"
list_name="100kb_all_none_pearson"

dataset=${assembly}"_"${release}  # ex: hg38_2018-10

# export ASSAY_LIST='["h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3"]' # as json


# --- Check used directories ---

# set -e, in case check_dir fails, to stop bash script (not go to prediction)
set -e

# New folders created if the ones written are not existant. Be careful with model loading.
model="${output_path}/2021-epiatlas/hg38_100kb_all_none_pearson/assay_1l_3000n"
log="${model}/predict_IHEC_2019-11"

program_path="${gen_path}/sources/epi_ml/epi_ml"
cd ${program_path}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py ${log}"
python ${program_path}/python/utils/check_dir.py ${log}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py --exists ${model}"
python ${program_path}/python/utils/check_dir.py --exists ${model}


# --- Creating correct paths for epilap and launching ---
timestamp=$(date +%s)

arg2="${input_path}/hparams/${hparams}"
arg3="${input_path}/hdf5_list/${dataset}/${list_name}.list"
arg4="${input_path}/chromsizes/${assembly}.noy.chrom.sizes"
arg5="${input_path}/metadata/${dataset}_final.json"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"


printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/main.py --predict $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} --model ${model} > ${out1} 2> ${out2}"
python ${program_path}/python/main.py --predict $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} --model ${model} > ${out1} 2> ${out2}


categories="dataset_name reference_registry_id releasing_group"
to_augment="${log}/test_prediction.csv"
metadata="${arg5}"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/augment_predict_file.py ${to_augment} ${metadata} ${categories} >> ${out1} 2>> ${out2}"
python ${program_path}/python/augment_predict_file.py ${to_augment} ${metadata} ${categories} >> ${out1} 2>> ${out2}
