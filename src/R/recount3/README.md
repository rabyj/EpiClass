
## 1) Download recount3 metadata

recount3 contains n=316,443 samples and n=8,742 projects. Each project has 5 metadata files: proj_meta, recount_project, recount_qc, recount_seq_qc, pred. The recount3 R package download the metadata and a custom script combine the 5 metadata tables into one with 175 columns for a given project.

Notes: For most projects, metadata were complete (n=8,311). For 366 projects, metadata were incomplete, all columns from the "pred" metadata file were missing, these tables had 168 columns. Missing fields were filled with NAs. Few samples were missing (from multiple projects), metadata files were individually downloaded to add theses samples. Final metadata file has 316,440 samples. The 3 missing samples came from the same project SRP101844 (SRR5341595, SRR5341596, SRR5341597).

```bash
Rscript fetch_list_recount3.R human
awk '$3=="sra"' human/available_projects.tsv | cut -f1 > human/sra_projects.tsv
Rscript fetch_metadata_recount3.R human human/sra_projects.tsv

# merge complete metadata and filled empty field with NAs
cat ~/scratch/recount3_human/complete/recount3* | tr ' ' '_' |  sort -rk1 | uniq | awk -F'\t' 'BEGIN {OFS="\t"} {for (i=1; i<=NF; i++) if ($i == "") $i = "NA"; print}' > human/merged_metadata_part1.tsv
# n = 303,048

# merge incomplete metadata
head -1 human/merged_metadata_part1.tsv | tr '\t' '\n' > human/header.txt
ls ~/scratch/recount3_human/incomplete/recount3_human_* > human/list_of_incomplete.txt
Rscript combine_incomplete.R human human/header.txt human/list_of_incomplete.txt
# n = 11,643

cat ~/scratch/recount3_human/incomplete/recount3* | tr ' ' '_' |  sort -rk1 | uniq > human/merged_metadata_part2_with_missing_fields.tsv
awk '{print $0"\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t"$168}' human/merged_metadata_part2_with_missing_fields.tsv | cut -f 1-167,169- > human/merged_metadata_part2.tsv

tail -n+2 human/merged_metadata_part2.tsv | cat human/merged_metadata_part1.tsv - | sort -rk1 > human/merged_metadata_part1_and_2.tsv
#n = 314,690

# missing samples
# list missing samples and create URL to download metadata files of projects with missing samples
cut -f2 human/merged_metadata_part1_and_2.tsv | tail -n+2 | sort -k1b,1 > human/samples_downloaded.txt

awk '$4=="sra"' human/available_samples.tsv | sort -k1b,1 | join -v1 - human/samples_downloaded.txt | tr ' ' '\t' | sort -k2 | awk '{OFS="\t"; suf=substr($2, length($2)-1, 2); print $1,$2,suf,"http://duffel.rail.bio/recount3/human/data_sources/sra/base_sums/"suf"/"$2"/"substr($1, length($1)-1, 2)"/sra.base_sums."$2"_"$1".ALL.bw"}' > human/missing_samples.txt

# url for recount_project metadata
awk 'BEGIN {print "#!/bin/bash"} {print "wget http://duffel.rail.bio/recount3/human/data_sources/sra/metadata/"$3"/"$2"/sra.sra."$2".MD.gz"}' human/missing_samples.txt | uniq > human/url_missing_metadata_sra.sh

# url for recount_qc metadata
awk 'BEGIN {print "#!/bin/bash"} {print "wget http://duffel.rail.bio/recount3/human/data_sources/sra/metadata/"$3"/"$2"/sra.recount_qc."$2".MD.gz"}' human/missing_samples.txt | uniq > human/url_missing_metadata_recount_qc.sh

# url for recount_qc metadata
awk 'BEGIN {print "#!/bin/bash"} {print "wget http://duffel.rail.bio/recount3/human/data_sources/sra/metadata/"$3"/"$2"/sra.recount_seq_qc."$2".MD.gz"}' human/missing_samples.txt | uniq > human/url_missing_metadata_recount_seq_qc.sh

bash human/url_missing_metadata_sra.sh
bash human/url_missing_metadata_recount_qc.sh
bash human/url_missing_metadata_recount_seq_qc.sh

Rscript complete_metadata.R human
cat ~/scratch/recount3_human/missing_formated/recount3* | tr ' ' '_' |  sort -rk1 | uniq | awk -F'\t' 'BEGIN {OFS="\t"} {for (i=1; i<=NF; i++) if ($i == "") $i = "NA"; print}' > human/merged_metadata_part3.tsv
cat human/merged_metadata_part1_and_2.tsv human/merged_metadata_part3.tsv | sort -rk1 | uniq > human/complete_metadata.tsv
```

## 2) Extract and harmonize metadata

Input file: complete_metadata.tsv
Sample ID is stored in the second column names "external_id".
The column *sra.sample_attributes* (#24) is the field with most information valuable to EpiClass. The field is arrange as a dictionnary using the format key;;value (e.g "age;;38"). The multiple key-value pairs are separated by "|" character. Keys to include from *sra.sample_attributes* were manually curated and writing to *sra.sample_attributes/sra.sample_attributes.included_keys_list.txt* file.
These are the other fields used for metadata extraction in recount3: sra.design_description, sra.sample_description, sra.sample_title, recount_pred.curated.type, recount_pred.pattern.predict.type and recount_pred.pred.type

Keywords dictionary were manually created according to recount3 metadata
 - keywords_assay.tsv
 - keywords_sex.tsv
 - keywords_cancer.tsv
 - keywords_lifestage.tsv
 - keywords_biomat.tsv
 - keywords_biospecimen.tsv

Input file with EpiClass predictions: recount3_merged_preds_harmonized_metadata_20250122_leuk2.tsv.gz

```bash
### extract prediction from EpiClass
zcat ../recount3_merged_preds_harmonized_metadata_20250122_leuk2.tsv.gz | tr ' ' '_' | cut -f 2,4,5,18,19,24,25,29,30,37,38 | sed 's/Predicted_class_(/predicted_/g' | sed 's/Max_pred_(/score_/g' | sed 's/harmonized_//g' | sed 's/donor_//g' | sed 's/)//g' | sed 's/sample_cancer_high/cancer/g' | sed 's/_epiclass//g' | sed 's/life_stage/lifestage/g' | sed 's/biomaterial_type/biomat/g' > EpiClass_predictions.tsv

### extract and melt "sra.sample_attributes" field (column #24)
mkdir sra.sample_attributes
cut -f2,24 ../complete_metadata.tsv | tr '|' '\t' | awk '{for (i=2; i<=NF; i+=1) {split($i,n,";;"); print $1"\t"n[1]"\t"tolower(n[2])}}' | awk 'NR == FNR {keywords[$1]=1; next} {for (term in keywords) {if($2==term) {print; next}}}' sra.sample_attributes/sra.sample_attributes.included_keys_list.txt - | grep -wvf sra.sample_attributes/unavailable_terms_list.txt - | awk 'NF==3' > sra.sample_attributes/sra.sample_attributes.included_keys.melted.tsv

### curate "sra.sample_attributes" field
Rscript sra.sample_attributes/curate_sra.sample_attributes.R
### produce *recount_combined_informative_fields.tsv*

### extract metadata for each categorie
Rscript extract_metadata.freeze1.R recount_combined_informative_fields.tsv freeze1
### produce *metadata.freeze1.tsv* and *full_info_metadata.freeze1.tsv*
```
