#!/bin/bash
#SBATCH --time=:::time:::
#SBATCH --account=:::your-account:::
#SBATCH --job-name==:::your-job-name:::
#SBATCH --output=./slurm_files/%x-job%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=:::memory::: # ex: 16G
#SBATCH --mail-user=john.doe@domain.com
#SBATCH --mail-type=END,FAIL

# NOTE: The values in between ':::' are to be replaced by the user

# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

export PYTHONUNBUFFERED=TRUE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print =========================================="
fi

gen_path="/path/to/epiclass" # MODIFY, input/output directories
input_path="${gen_path}/epiclass/input"
output_path="${gen_path}/epiclass/output/logs"

gen_program_path="${gen_path}/sources/epiclass" # MODIFY: git root
program_path="${gen_program_path}/src/python/epiclass"

slurm_out_folder="${gen_path}/epiclass/output/sub/slurm_files"

for path in ${slurm_out_folder} ${gen_program_path} ${input_path} ${output_path}; do
  if [ ! -d ${path} ]; then
    echo "${path} is not a directory. Please check the path."
    exit 1
  else
    echo "Used directory: ${path}"
  fi
done


# --- use correct environment ---

set -e
if [[ -n "$SLURM_JOB_ID" ]]; then
  cd $SLURM_TMPDIR
  bash ${gen_program_path}/src/bash_utils/setup_venv.sh -r ${gen_program_path}/requirements/minimal_requirements.txt -s ${gen_program_path}/src/python &>${slurm_out_folder}/${SLURM_JOB_ID}_setup.log
  source epiclass_env/bin/activate
else
  source /path/to/preinstalled/venv/bin/activate # MODIFY
fi


# --- choose category + hparams + source files ---

# MODIFY THINGS HERE

# RESTORE="--restore" # COMMENT IF TRAINING # IMPORTANT
# NO_VALID="hell yeah" # COMMENT IF 10fold  TRAINING # IMPORTANT

category="assay_epiclass"

export EXCLUDE_LIST='["other", "--", "NA", "", "unknown"]'
export MIN_CLASS_SIZE="10" # IMPORTANT

hparams="human_longer_oversample" # IMPORTANT

release="epiatlas-dfreeze-v2.1"
assembly="hg38"

resolution="100kb"
basename="${resolution}_all_none"
list_name="${basename}" # IMPORTANT

export LAYER_SIZE="3000" # IMPORTANT
export NB_LAYER="1"

log="${output_path}/${release}/${assembly}_${basename}/${category}_${NB_LAYER}l_${LAYER_SIZE}n" # IMPORTANT# IMPORTANT# IMPORTANT# IMPORTANT
log="${log}/10fold-oversampling"


# --- Creating correct paths for programs/launching ---

timestamp=$(date +%s)

hparams="${input_path}/hparams/${hparams}.json"
hdf5_list="${input_path}/hdf5_list/hg38_epiatlas-freeze-v2/${list_name}.list"
chroms="${input_path}/chromsizes/hg38.noy.chrom.sizes"
metadata="${input_path}/metadata/dfreeze-v2/hg38_2023-epiatlas-dfreeze-pospurge-nodup.json"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

set -e
for path in ${hparams} ${hdf5_list} ${chroms} ${metadata}; do
  if [ ! -f ${path} ]; then
    echo "${path} is not a file. Please check the path."
    exit 1
  else
    echo "Input: ${path}"
  fi
done


# --- Pre-checks ---

cd ${program_path}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/check_dir.py ${log}"
python ${program_path}/utils/check_dir.py ${log}

# Preconditions passed, copy launch script to log dir.
if [[ -n "$SLURM_JOB_ID" ]]; then
  scontrol write batch_script ${SLURM_JOB_ID} ${log}/launch_script_${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.sh
fi


# --- MAIN PROGRAM ---

echo "Time before launch: $(date +%F_%T)"
printf '\n%s\n' "Launching following command"
if [[ -n "$NO_VALID" ]]; then #if variable exists
  # --- complete training without validation set launch ---
  if [[ "$log" == *"10fold"* ]]; then
    log="$log/notactually10foldbaka"
    printf '\n%s\n' "Incoherent log path, changing log to $log"
  fi

  printf '%s\n' "python ${program_path}/epiatlas_training_no_valid.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} > ${out1} 2> ${out2}"
  python ${program_path}/epiatlas_training_no_valid.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} >"${out1}" 2>"${out2}"
  echo "Time after launch: $(date +%F_%T)"
  exit

elif [[ -n "$RESTORE" ]]; then
  # --- kfold launch ---
  printf '%s\n' "python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} --restore > ${out1} 2> ${out2}"
  python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} --restore >"${out1}" 2>"${out2}"
  exit

else
  # --- kfold launch ---
  printf '%s\n' "python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} > ${out1} 2> ${out2}"
  python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} >"${out1}" 2>"${out2}"
fi
echo "Time after launch: $(date +%F_%T)"


# --- More logging ---
set +e

cd ${log}
printf '\n%s\n' "Launching following command"
printf '%s\n' "cat split*/validation_prediction.csv | sort -ru > full-10fold-validation_prediction.csv"
cat split*/validation_prediction.csv | sort -ru >full-10fold-validation_prediction.csv

to_augment="${log}/full-10fold-validation_prediction.csv"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories

printf '%s\n' "python ${program_path}/utils/create_confusion_matrices.py --from_prediction ${to_augment}"
python ${program_path}/utils/create_confusion_matrices.py --from_prediction ${to_augment}

# Copy slurm output file to log dir
if [[ -n "$SLURM_JOB_ID" ]]; then
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${log}/
fi
