CHROMS=("chr10" "chr11" "chr12" "chr13" "chr15" "chr16" "chr17" "chr18" "chr19" "chr1" "chr20" "chr21" "chr22" "chr2" "chr3" "chr4" "chr5" "chr6" "chr7" "chr8" "chr9" "chrX")

# create sbatch files
for chrom in "${CHROMS[@]}"; do

    temp_file="${chrom}_1.sh.temp"
    filename1="assay_${chrom}_1.sh"
    filename2="cell_type_${chrom}_1.sh"

    sed "s@--CHROM--@${chrom}@g" sub_template.sh > ${temp_file}

    sed "s@--CATEGORY--@assay@g" ${temp_file} > ./launch_epiML/${filename1}
    sed "s@--CATEGORY--@cell\_type@g" ${temp_file} > ./launch_epiML/${filename2}

    rm ${temp_file}

done

# create log directories
# for chrom in "${CHROMS[@]}"; do

#     log_dir="/lustre03/project/6007017/rabyj/epi_ml_project/sub/logs/2018-10/hg19_100kb_${chrom}_1_none_pearson"
#     mkdir "${log_dir}"
#     mkdir "${log_dir}/major_ct_assay"
#     mkdir "${log_dir}/major_ct_cell_type"

# done
