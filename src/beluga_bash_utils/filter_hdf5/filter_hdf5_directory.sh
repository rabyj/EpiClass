#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-jacquesp
#SBATCH --job-name=filter_chr14
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-user=joanny.raby@usherbrooke.ca
#SBATCH --mail-type=END

FRACTION="0.1"

project_path="/lustre03/project/6007017/rabyj/epi_ml_project"
hdf5_path="/lustre04/scratch/rabyj/local_ihec_data/2018-10/hg19/hdf5"

. ${project_path}/sub/epigeec_venv/bin/activate

source_folder="${hdf5_path}/100kb_all_none"
out_folder="${hdf5_path}/100kb_assay_${FRACTION}_none"

chroms="${project_path}/chromsizes/hg19.noy.chrom.sizes"
filter="${project_path}/filter/hg19.assay_${FRACTION}.bed"


for hdf5_file in ${source_folder}/*_value.hdf5; do

    old_basename=$(basename ${hdf5_file})
    split_basename=(${old_basename//_/ })
    new_basename="${split_basename[0]}_100kb_assay_${FRACTION}_none_value.hdf5"
    new_hdf5="${out_folder}/${new_basename}"

    # echo ${hdf5_file}
    # echo $part1
    # echo $part2
    # echo ${split_basename[0]}
    # echo ${new_basename}
    # echo ${new_hdf5}

    echo "epigeec filter --select ${filter} ${hdf5_file} ${chroms} ${new_hdf5}"
    epigeec filter --select ${filter} ${hdf5_file} ${chroms} ${new_hdf5}
    chown rabyj:def-jacquesp ${new_hdf5}

done

# i=0
# for hdf5_file in ${source_folder}/*_value.hdf5; do
#     if [ $i -gt 4800 ]
#     then
#         old_basename=$(basename ${hdf5_file})
#         split_basename=(${old_basename//_/ })
#         new_basename="${split_basename[0]}_100kb_assay_${FRACTION}_none_value.hdf5"
#         new_hdf5="${out_folder}/${new_basename}"

#         echo "epigeec filter --select ${filter} ${hdf5_file} ${chroms} ${new_hdf5}"
#         epigeec filter --select ${filter} ${hdf5_file} ${chroms} ${new_hdf5}
#         chown rabyj:def-jacquesp ${new_hdf5}
#     fi
#     i=$((i + 1))
# done
