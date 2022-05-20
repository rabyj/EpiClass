#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=def-jacquesp
#SBATCH --job-name=2step_--ASSAY--
#SBATCH --output=../logs/2018-10/hg19_100kb_all_none_pearson/2step_1l_3000n/assay_--ASSAY--/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=186G
#SBATCH --mail-user=joanny.raby@usherbrooke.ca
#SBATCH --mail-type=END

project_path="/lustre03/project/6007017/rabyj/epi_ml_project"

# chose category + hparams + source files
category="cell_type"
hparams="human_base.json"
release="2018-10"
assembly="hg19"
list_name="100kb_all_none_pearson"

export LAYER_SIZE="3000"
export NB_LAYER="1"
export STEP1_ASSAY="--ASSAY--"

log_dir="${release}/${assembly}_${list_name}/2step_1l_3000n/assay_--ASSAY--"

# check log dir + make sure permissions are good
chown -R rabyj:def-jacquesp "${project_path}/sub/logs/${log_dir}"
chmod -R g+s "${project_path}/sub/logs/${log_dir}"

sh ${project_path}/sub/epi_ml.sh ${category} ${hparams} ${release} ${assembly} ${list_name} ${log_dir}
