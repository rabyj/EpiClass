#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=your-account
#SBATCH --job-name=EA-CT-k10-10kb
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
source /lustre07/scratch/rabyj/envs/epilap/bin/activate

# --- choose category + hparams + source files ---
# MODIFY THINGS HERE

category="cell_type" # IMPORTANT
# export ASSAY_LIST='["h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3", "input", "rna_seq", "mrna_seq", "wgbs"]' # as json
# export ASSAY_LIST='["female", "male"]'
export EXCLUDE="other"

hparams="human_10kb_l2.json" # IMPORTANT
# hparams="human_base.json" # IMPORTANT
release="2022-epiatlas"
assembly="hg38"
resolution="10kb" # IMPORTANT

basename="${resolution}_all_none"
list_name="${basename}_plus" # IMPORTANT

dataset=${assembly}"_"${release} # ex: hg38_2018-10

export LAYER_SIZE="1500" # IMPORTANT
export NB_LAYER="1"

log="${output_path}/${release}/${assembly}_${basename}_pearson/${category}_${NB_LAYER}l_${LAYER_SIZE}n" # IMPORTANT# IMPORTANT# IMPORTANT# IMPORTANT
# log="${log}/10fold" # IMPORTANT
log="${log}/10fold-l2" # IMPORTANT

# --- Creating correct paths for epilap and launching ---

timestamp=$(date +%s)

arg2="${input_path}/hparams/${hparams}"
arg3="${input_path}/hdf5_list/${dataset}/${list_name}.list"
arg4="${input_path}/chromsizes/hg38.noy.chrom.sizes"
arg5="${input_path}/metadata/merged_EpiAtlas_allmetadatav9.json"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

program_path="${gen_path}/sources/epi_ml/epi_ml"
cd ${program_path}

set -e
printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/utils/check_dir.py ${log}"
python ${program_path}/python/utils/check_dir.py ${log}


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


# # --- kfold launch ---
printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/python/epiatlas_training.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > ${out1} 2> ${out2}"
python ${program_path}/python/epiatlas_training.py $category ${arg2} ${arg3} ${arg4} ${arg5} ${log} > ${out1} 2> ${out2}


# # categories="data_generating_centre epirr_id uuid track_type"
# to_augment="${log}/full-10fold-validation_prediction.csv"
# # to_augment="${log}/validation_prediction.csv"
# metadata="${arg5}"

# printf '\n%s\n' "Launching following command"
# printf '%s\n' "python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
# python ${program_path}/python/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories
