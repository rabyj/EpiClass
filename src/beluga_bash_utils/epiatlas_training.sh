#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=your-account
#SBATCH --job-name=your-job-name
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --mail-user=john.doe@domain.ca
#SBATCH --mail-type=END,FAIL

# ----> TEMPLATE PATHS TO MODIFY FOR SURE ARE IN []. Use regex to find, e.g. \[.*\]<----

# Path of current file, for input logging purposes
SCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=TRUE

if [ X"$SLURM_STEP_ID" = "X" ] && [ X"$SLURM_PROCID" = "X"0 ]
then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print =========================================="
fi

gen_path="/home/[username]/[general-workspace]"
input_path="${gen_path}/[project_folder]/input"
output_path="${gen_path}/[project_folder]/output/logs"

# use correct environment
source [/path/to/venv]

# --- choose category + hparams + source files ---
# MODIFY THINGS HERE

category="cell_type" # IMPORTANT
# export ASSAY_LIST='["h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3", "input", "rna_seq", "mrna_seq", "wgbs"]' # as json
export LABEL_LIST='["female", "male"]'
# export LABEL_LIST='["single_end", "paired_end"]'
# export EXCLUDE_LIST='["other", "--", "NA", ""]'


hparams="human_longer.json" # IMPORTANT
release="2023-01-epiatlas-freeze"
assembly="hg38"
resolution="100kb" # IMPORTANT


basename="${resolution}_all_none"
list_name="${basename}" # IMPORTANT

dataset=${assembly}"_"${release} # ex: hg38_2018-10

echo $dataset

export LAYER_SIZE="3000" # IMPORTANT
export NB_LAYER="1"

log="${output_path}/${release}/${assembly}_${basename}/${category}_${NB_LAYER}l_${LAYER_SIZE}n" # IMPORTANT# IMPORTANT# IMPORTANT# IMPORTANT
log="${log}/10fold" # IMPORTANT

# --- Creating correct paths for programs/launching ---

timestamp=$(date +%s)

arg2="${input_path}/hparams/${hparams}"
arg3="${input_path}/hdf5_list/${dataset}/${list_name}.list"
arg4="${input_path}/chromsizes/hg38.noy.chrom.sizes"
arg5="${input_path}/metadata/metadata_file.json"  # IMPORTANT
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

set -e
echo "Input arguments:"
for var in $arg2 $arg3 $arg4 $arg5
do
ls $var
done


# --- Pre-checks ---

program_path="${gen_path}/[path-to-program-python-scripts]"
cd ${program_path}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py ${log}"
python ${program_path}/python/utils/check_dir.py ${log}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/preconditions.py -m ${arg5}"
python ${program_path}/python/utils/preconditions.py -m ${arg5}

cp -v "$SCRIPTPATH" "$log/launch_script_${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.sh"


# --- Transfer files to node scratch ---

# if [ X"$SLURM_STEP_ID" = "X" ] && [ X"$SLURM_PROCID" = "X"0 ]
# then
#   newdir="$SLURM_TMPDIR/hdf5s/"
#   mkdir $newdir
#   filelist=$(cat ${arg3})
#     for FILE in ${filelist}
#     do
#       cp -L ${FILE} ${newdir}
#     done
# fi


if [ X"$SLURM_STEP_ID" = "X" ] && [ X"$SLURM_PROCID" = "X"0 ]
then
  project="[/path/to/project]"
  tar_file="${project}/input/hdf5/epiatlas_dfreeze_${resolution}_all_none.tar"  # IMPORTANT

  cd $SLURM_TMPDIR

  echo "Untaring $tar_file in $SLURM_TMPDIR"
  time tar -xf $tar_file

  export HDF5_PARENT="epiatlas_dfreeze_${resolution}_all_none" # IMPORTANT
fi


# --- no valid launch ---

if [[ -n "$NO_VALID" ]] #if variable exists
then
  if [[ "$log" == *"10fold"* ]]; then
    log="$log/notactually10fold"
    printf '\n%s\n' "Incoherent log path, changing log to $log"
  fi
  echo "Time before launch: $(date +%F_%T)"
  printf '\n%s\n' "Launching following command"
  printf '%s\n' "python ${program_path}/python/epiatlas_training_no_valid.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > ${out1} 2> ${out2}"
  python ${program_path}/python/epiatlas_training_no_valid.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > "${out1}" 2> "${out2}"
  echo "Time after launch: $(date +%F_%T)"
  exit
else
  # --- kfold launch ---
  echo "Time before launch: $(date +%F_%T)"
  printf '\n%s\n' "Launching following command"
  if [[ -n "$RESTORE" ]]
  then
    printf '%s\n' "python ${program_path}/python/epiatlas_training.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} --restore > ${out1} 2> ${out2}"
    python ${program_path}/python/epiatlas_training.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} --restore > "${out1}" 2> "${out2}"
    exit
  else
    printf '%s\n' "python ${program_path}/python/epiatlas_training.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > ${out1} 2> ${out2}"
    python ${program_path}/python/epiatlas_training.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > "${out1}" 2> "${out2}"
  fi
  echo "Time after launch: $(date +%F_%T)"
fi


# --- More logging ---

export LOG="${log}"
export NO_TRUE="False"

cd ${log}
printf '\n%s\n' "Launching following command"
printf '%s\n' "cat split*/validation_prediction.csv | sort -ru > full-10fold-validation_prediction.csv"
cat split*/validation_prediction.csv | sort -ru > full-10fold-validation_prediction.csv

# categories="spam bam foo"
to_augment="${log}/full-10fold-validation_prediction.csv"
metadata="${arg5}"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories


# Copy slurm output file to log dir
if [ X"$SLURM_STEP_ID" = "X" ] && [ X"$SLURM_PROCID" = "X"0 ]
then
  slurm_out_folder="${gen_path}/[path-to-slurm-output]"
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${log}/
fi
