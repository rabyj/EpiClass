#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-jacquesp
#SBATCH --job-name=--CHROM--_--CATEGORY--
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --mail-user=joanny.raby@usherbrooke.ca
#SBATCH --mail-type=END
#SBATCH --output=./%x-%j.out

# --- WARNING: THE LOG DIRECTORY USED MUST ALREADY EXIST ---

project_path="/lustre03/project/6007017/rabyj/epi_ml_project"

# chose category + hparams + source files
category="--CATEGORY--"
hparams="human_base.json"
release="2018-10"
assembly="hg19"
list_name="100kb_--CHROM--_1_none_pearson"

export LAYER_SIZE="3000"
export NB_LAYER="1"

log_dir="${release}/${assembly}_${list_name}/major_ct_--CATEGORY--"

# check log dir + make sure permissions are good
log="${project_path}/sub/logs/${log_dir}"
chown -R rabyj:def-jacquesp "${log}"
chmod -R g+s "${log}"

timestamp=$(date +%s)
sh ${project_path}/sub/epi_ml.sh ${category} ${hparams} ${release} ${assembly} ${list_name} ${log_dir} > "${log}/output_${timestamp}.o" 2> "${log}/output_${timestamp}.e"
