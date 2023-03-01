#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=your-account
#SBATCH --job-name=your-job-name
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --mail-user=john.doe@domain.ca
#SBATCH --mail-type=END,FAIL


# ----> TEMPLATE PATHS TO MODIFY FOR SURE ARE IN []. Use regex to find, e.g. \[.*\]<----

export PYTHONUNBUFFERED=TRUE

SCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

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

# array=( "LGBM" "LR" "RF" "LinearSVC" )
mode="predict" # IMPORTANT #predict, tune or both
models="all" # IMPORTANT

category="harmonized_EpiRR_status" # IMPORTANT
# export EXCLUDE_LIST='["others"]'
# export LABEL_LIST='["mixed", "unknown"]' # single quotes give litteral string
# export ASSAY_LIST='["h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3", "input", "rna_seq", "mrna_seq", "wgbs"]' # as json

export MIN_CLASS_SIZE="10" # IMPORTANT


release="2023-01-epiatlas-freeze"
assembly="hg38"
resolution="100kb" # IMPORTANT

basename="${resolution}_all_none"
list_name="${basename}" # IMPORTANT

dataset=${assembly}"_"${release}  # ex: hg38_2018-10

log_base="${output_path}/${release}/${assembly}_${basename}/${category}" # IMPORTANT# IMPORTANT#
log="${log_base}/predict-10fold" # IMPORTANT
echo "Expected log: ${log}"


# --- Creating correct paths for programs/launching ---

timestamp=$(date +%s)

hparams="${input_path}/hparams/estimators_assay_100kb.json"

list="${input_path}/hdf5_list/${dataset}/${list_name}.list"
chrom="${input_path}/chromsizes/hg38.noy.chrom.sizes"
meta="${input_path}/metadata/[metadata_file.json]"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

set -e
echo "Input arguments:"
for var in $list $chrom $meta $hparams
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
printf '%s\n' "python ${program_path}/python/utils/preconditions.py -m ${meta}"
python ${program_path}/python/utils/preconditions.py -m ${meta}

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


# if [ X"$SLURM_STEP_ID" = "X" ] && [ X"$SLURM_PROCID" = "X"0 ]
# then
#   project="[/path/to/project]"
#   tar_file="${project}/input/hdf5/epiatlas_dfreeze_${resolution}_all_none.tar"  # IMPORTANT

#   cd $SLURM_TMPDIR

#   echo "Untaring $tar_file in $SLURM_TMPDIR"
#   time tar -xf $tar_file

#   export HDF5_PARENT="epiatlas_dfreeze_${resolution}_all_none" # IMPORTANT
# fi


# --- launch ---
echo "Time before launch: $(date +%F_%T)"
if [ "${mode}" = "tune" ]
then
  export CONCURRENT_CV="1" # IMPORTANT
  n_iter="15" # IMPORTANT
  printf '\n%s\n' "Launching following command"
  printf '%s\n' "python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --tune --models ${models} -n ${n_iter} > ${out1} 2> ${out2}"
  python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --tune --models ${models} -n ${n_iter} > ${out1} 2> ${out2}
fi


if [ "${mode}" = "predict" ]
then
  printf '\n%s\n' "Launching following command"
  printf '%s\n' "python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --predict --models ${models} --hyperparams ${hparams} > ${out1} 2> ${out2}"
  python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --predict --models ${models} --hyperparams ${hparams} > ${out1} 2> ${out2}
fi


if [ "${mode}" = "both" ]
then
  export CONCURRENT_CV="2" # IMPORTANT
  n_iter="20" # IMPORTANT
  printf '\n%s\n' "Launching following command"
  printf '%s\n' "python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --full-run -n ${n_iter} > ${out1} 2> ${out2}"
  python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --full-run -n ${n_iter} > ${out1} 2> ${out2}
fi

if [ "${mode}" = "predict-new" ]
then
  printf '\n%s\n' "Launching following command"
  printf '%s\n' "python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --predict-new --models ${models} > ${out1} 2> ${out2}"
  python ${program_path}/python/other_estimators.py ${category} ${list} ${chrom} ${meta} ${log} --predict-new --models ${models} > ${out1} 2> ${out2}
  wait
  exit
fi
echo "Time after launch: $(date +%F_%T)"


# -- Post processing output --

# BEWARE, MIGHT HAVE TO MODIFY

# cd ${log}
# array=( "LGBM" "LR" "RF" "LinearSVC" )
# for model in "${array[@]}"; do
#   output="${model}_full-10fold-validation_prediction.csv"
#   cat ${log}/${model}/${model}_split*_validation_prediction.csv | sort -ru > ${output}

#   printf '%s\n' "python ${program_path}/python/utils/augment_predict_file.py ${output} ${meta} --all-categories"
#   python ${program_path}/python/utils/augment_predict_file.py ${output} ${meta} --all-categories
# done
# wait

# printf '%s\n' "python ${program_path}/python/utils/merge_all_predictions.py ${log_base}_1l_3000n/10fold/full-10fold-validation_prediction_augmented-all.csv ${log}/*full*validation_prediction_augmented-all.csv"
# python ${program_path}/python/utils/merge_all_predictions.py ${log_base}_1l_3000n/10fold/full-10fold-validation_prediction_augmented-all.csv ${log}/*full*validation_prediction_augmented-all.csv


# Copy slurm output file to log dir
if [ X"$SLURM_STEP_ID" = "X" ] && [ X"$SLURM_PROCID" = "X"0 ]
then
  slurm_out_folder="${gen_path}/[path-to-slurm-output]"
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${log}/
fi
