#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --account=def-jacquesp
#SBATCH --job-name=variance
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-user=joanny.raby@usherbrooke.ca
#SBATCH --mail-type=END

project_path="/lustre03/project/6007017/rabyj/epi_ml_project"
script_path="${project_path}/epi_ml/epi_ml/python/compute_bin_variance.py"

. ${project_path}/epi_ml/venv/bin/activate

release="2018-10"
assembly="hg19"
signal_type="100kb_all_none_pearson"
list_name="${signal_type}_major_ct"

name=${assembly}"_"${release} # ex: hg38_2018-10

hdf5_list=${name}"/"${list_name}".list"
chrom_file=${assembly}".noy.chrom.sizes"
json=${name}"_final.json"

log_dir="${release}/${assembly}_${signal_type}/variance_major_cell_type"

hdf5_list_path="${project_path}/hdf5_list/${hdf5_list}"
chrom_file_path="${project_path}/chromsizes/${chrom_file}"
json_path="${project_path}/metadata/${json}"
log_dir_path="${project_path}/sub/logs/${log_dir}"

# echo ${hdf5_list_path}
# echo ${chrom_file_path}
# echo ${json_path}
# echo ${log_dir_path}

python ${script_path} ${hdf5_list_path} ${chrom_file_path} ${json_path} ${log_dir_path}
