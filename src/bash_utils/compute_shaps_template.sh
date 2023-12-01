#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=:::account:::
#SBATCH --job-name=:::job_name:::
#SBATCH --output=/lustre06/project/6007017/rabyj/epilap/output/sub/slurm_files/%x-job%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --mail-user=:::email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC2317  # Don't warn about unreachable commands in this file
# shellcheck disable=SC2086  # Don't warn about double quoting
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

export PYTHONUNBUFFERED=TRUE

if [[ -n "$SLURM_JOB_ID" ]];
then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print =========================================="
fi

gen_path="/lustre06/project/6007017/rabyj"
input_path="${gen_path}/epilap/input"
gen_program_path="${gen_path}/sources/epi_ml"
program_path="${gen_program_path}/src/python/epi_ml"

slurm_out_folder="${gen_path}/epilap/output/sub/slurm_files"

# use correct environment
set -e
if [[ -n "$SLURM_JOB_ID" ]];
then
  cd $SLURM_TMPDIR
  bash ${gen_program_path}/src/bash_utils/setup_venv.sh -r ${gen_program_path}/requirements/minimal_requirements.txt -s ${gen_program_path}/src/python &> ${slurm_out_folder}/${SLURM_JOB_ID}_setup.log
  source epiclass_env/bin/activate
else
  source /lustre07/scratch/rabyj/envs/epiclass/bin/activate
fi


# --- choose category + hparams + source files ---

model="NN" # IMPORTANT

category=:::category::: # IMPORTANT
resolution="100kb" # IMPORTANT

# -- NN --

# IMPORTANT!!!!
model_path=":::model_path:::"
output_log=":::output_log:::"

# --- Creating correct paths for programs/launching ---

timestamp=$(date +%s)

list_background=":::background_list:::"
list_explain=":::eval_list:::"

chroms="${input_path}/chromsizes/hg38.noy.chrom.sizes"
out1="${output_log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${output_log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

echo "Input arguments:"
for var in ${list_background} ${list_explain} ${chroms}
do
ls $var
done


# --- Pre-checks ---

program_path="${gen_path}/sources/epi_ml/src/python/epi_ml"
cd ${program_path}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/check_dir.py --exists ${model_path}"
python ${program_path}/utils/check_dir.py --exists ${model_path}

printf '%s\n' "python ${program_path}/utils/check_dir.py ${output_log}"
python ${program_path}/utils/check_dir.py ${output_log}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/preconditions.py"
python ${program_path}/utils/preconditions.py

# Preconditions passed, copy launch script to log dir.
if [[ -n "$SLURM_JOB_ID" ]];
then
scontrol write batch_script ${SLURM_JOB_ID} ${output_log}/launch_script_${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.sh
fi


# --- Transfer files to node scratch ---

if [[ -n "$SLURM_JOB_ID" ]];
then
  hdf5s_location="/lustre06/project/6007515/ihec_share/local_ihec_data/epiatlas/hg38/hdf5"
  name="epiatlas_dfreeze_${resolution}_all_none"
  tar_file="${hdf5s_location}/${name}.tar"  # IMPORTANT

  cd $SLURM_TMPDIR

  echo "Untaring $tar_file in $SLURM_TMPDIR"
  tar -xf $tar_file

  export HDF5_PARENT="${name}" # IMPORTANT
  cd $name
  scp ${hdf5s_location}/${name}-4ctl/* . #extra files in v2.1
fi


# --- MAIN PROGRAM ---
set +e

# usage: compute_shaps.py [-h] -m {NN,LGBM} --background_hdf5 background-hdf5 --explain_hdf5 explain-hdf5 --chromsize CHROMSIZE [-l LOGDIR] [-o --output-name]
#                        [--model_file model_file] [--model_dir MODEL_DIR]

basecmd="python ${program_path}/compute_shaps.py -m ${model} --background_hdf5 ${list_background} --explain_hdf5 ${list_explain} --chromsize ${chroms} -l ${output_log} -o explain_${category}"
basecmd="${basecmd} --model_dir ${model_path}"


echo "Time before launch: $(date +%F_%T)"
printf '\n%s\n' "Launching following command"
printf '%s\n' "${basecmd} > ${out1} 2> ${out2}"
${basecmd} > ${out1} 2> ${out2}
echo "Time after launch: $(date +%F_%T)"



# Copy slurm output file to log dir
if [[ -n "$SLURM_JOB_ID" ]];
then
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${output_log}/
fi
