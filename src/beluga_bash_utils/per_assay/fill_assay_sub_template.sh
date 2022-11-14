assays=("smrna_seq" "mrna_seq" "wgb_seq" "h3k9me3" "h3k36me3" "h3k27me3" "h3k4me3" "h3k4me1" "input" "rna_seq" "h3k27ac" "chromatin_acc")

for assay in "${assays[@]}"; do
    # echo $assay
    echo "sed "s@--ASSAY--@${assay}@g" sub_template.sh > ${assay}.sh"
    # sed "s@--ASSAY--@${assay}@g" sub_template.sh > ${assay}.sh
done
