#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=your-account
#SBATCH --job-name=predict-sex
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
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
# MODIFY THINGS HERE

category="sex"
release="2022-epiatlas"
assembly="hg38"
basename="100kb_all_none"
list_name="${basename}-unknown-sex" # IMPORTANT

dataset=${assembly}"_"${release}  # ex: hg38_2018-10

export LAYER_SIZE="3000" # IMPORTANT
export NB_LAYER="1"

# IMPORTANT # IMPORTANT # IMPORTANT
base_log="${output_path}/${release}/${assembly}_${basename}_pearson/${category}_${NB_LAYER}l_${LAYER_SIZE}n"

model="${base_log}/10fold/split0" # IMPORTANT
log="${base_log}/predict_unknown" # IMPORTANT

# --- Creating correct paths for epilap and launching ---

timestamp=$(date +%s)

hdf5_list="${input_path}/hdf5_list/${dataset}/${list_name}.list"
chromsizes="${input_path}/chromsizes/hg38.noy.chrom.sizes"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

program_path="${gen_path}/sources/epi_ml/epi_ml"
cd ${program_path}


# set -e, in case check_dir fails, to stop bash script (not go to prediction)
set -e

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py ${log}"
python ${program_path}/python/utils/check_dir.py ${log}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py --exists ${model}"
python ${program_path}/python/utils/check_dir.py --exists ${model}


if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  newdir="$SLURM_TMPDIR/hdf5s/"
  mkdir $newdir
  filelist=$(cat ${arg3})
    for FILE in ${filelist}
    do
      cp ${FILE} ${newdir}
    done
fi


# --- launch ---
# printf '\n%s\n' "Launching following command"
# printf '%s\n' "python ${program_path}/python/predict.py ${hdf5_list} ${chromsizes} ${log}  --model ${model} > ${out1} 2> ${out2}"
# python ${program_path}/python/predict.py ${hdf5_list} ${chromsizes} ${log}  --model ${model} > ${out1} 2> ${out2}

to_augment="${log}/test_prediction.csv"
metadata="${input_path}/metadata/${dataset}_harmonizedv8.json"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories
