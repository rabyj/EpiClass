project_path="/lustre03/project/6007017/rabyj/epi_ml_project"

. ${project_path}/epi_ml/venv/bin/activate

category=$1
hparams=$2
release=$3
assembly=$4
list_name=$5
log=$6

name=${assembly}"_"${release} # ex: hg38_2018-10

hdf5_list=${name}"/"${list_name}".list"
chrom_file=${assembly}".noy.chrom.sizes"
json=${name}"_final.json"

python ${project_path}/epi_ml/epi_ml/python/main.py $category ${project_path}/hparams/${hparams} ${project_path}/hdf5_list/${hdf5_list} ${project_path}/chromsizes/${chrom_file} ${project_path}/metadata/${json} ${project_path}/sub/logs/${log}
