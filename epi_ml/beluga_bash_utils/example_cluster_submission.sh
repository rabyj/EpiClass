#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=your-account
#SBATCH --job-name=profile-hg38
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=john.doe@domain.ca
#SBATCH --mail-type=END,FAIL

export PYTHONUNBUFFERED=TRUE

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

export LAYER_SIZE="3000"
export NB_LAYER="1"
# export REMOVE_ASSAY=""
# export SELECT_ASSAY="rna_seq"
# export ASSAY_LIST='["h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3"]' # as json

# --- check log dir ---
# e.g. 2019-11/hg38_100kb_all_none_pearson/assay_1l_3000n

# set -e, in case check_dir fails, to stop bash script (not go to prediction)
set -e

program_path="${gen_path}/sources/epi_ml/epi_ml"
cd ${program_path}

log="${output_path}/${release}/${assembly}_${list_name}/${category}_${NB_LAYER}l_${LAYER_SIZE}n"
printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py ${log}"
python ${program_path}/python/utils/check_dir.py ${log}


# --- Creating correct paths for epilap and launching ---
timestamp=$(date +%s)

arg2="${input_path}/hparams/${hparams}"
arg3="${input_path}/hdf5_list/${dataset}/${list_name}.list"
arg4="${input_path}/chromsizes/${assembly}.noy.chrom.sizes"
arg5="${input_path}/metadata/${dataset}_final.json"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/main.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > ${out1} 2> ${out2}"
python ${program_path}/python/main.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > ${out1} 2> ${out2}
