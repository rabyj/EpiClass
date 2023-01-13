#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --account=def-jacquesp
#SBATCH --job-name=--FILTER_NAME--
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --mail-user=joanny.raby@usherbrooke.ca
#SBATCH --mail-type=END

# --- WARNING: OUT HDF5 DIRECTORY MUST ALREADY EXIST ---

project_path="/lustre03/project/6007017/rabyj/epi_ml_project"
hdf5_path="/lustre04/scratch/rabyj/local_ihec_data/2018-10/hg19/hdf5"

. ${project_path}/sub/epigeec_venv/bin/activate

filter_name="--FILTER_NAME--"
resolution="100kb"

source_list="${project_path}/hdf5_list/hg19_2018-10/${resolution}_all_none_pearson_major_ct.list"
out_folder="${hdf5_path}/${resolution}_${filter_name}_none"

# Check out folder
# echo ${out_folder}
# ls ${out_folder}
# exit

chroms="${project_path}/chromsizes/hg19.noy.chrom.sizes"
filter="${project_path}/filter/hg19.${filter_name}.bed"
mapfile hdf5_list < ${source_list}

for hdf5_file in ${hdf5_list[@]}; do

    old_basename=$(basename ${hdf5_file})
    split_basename=(${old_basename//_/ })
    new_basename="${split_basename[0]}_${resolution}_${filter_name}_none_value.hdf5"
    new_hdf5="${out_folder}/${new_basename}"

    echo "epigeec filter --select ${filter} ${hdf5_file} ${chroms} ${new_hdf5}"
    epigeec filter --select ${filter} ${hdf5_file} ${chroms} ${new_hdf5}

done

