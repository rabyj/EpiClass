
Input files *CA_metadata_4DB+all_pred.20240606_mod3.0.tsv* provided by Joanny R.
All metadata were extracted from on single field name "Metadata" (column #95). Keys to include from "Metadata" were manually curated and writing to *metadata_field/included_keys_list.txt* file

Keywords dictionary were manually created according to ChIP-Atlas metadata
 - keywords_sex.tsv
 - keywords_cancer.tsv
 - keywords_lifestage.tsv
 - keywords_biomat.tsv
 - keywords_biospecimen.tsv

```bash
### extract prediction from EpiClass
cut -f 1,7,9,10,27,28,41,42,55,56,129,130 CA_metadata_4DB+all_pred.20240606_mod3.0.tsv | sed 's/Predicted_class/predicted/g' | sed 's/Max_pred/score/g' | sed 's/assay7/assay/g' | sed 's/donorlife/lifestage/g' | sed 's/manual_target_consensus/expected_assay/g' > EpiClass_predictions_20240606_mod3.tsv

### extract and melt "Metadata" field (column #95)
mkdir metadata_field
cut -f1,95 CA_metadata_4DB+all_pred.20240606_mod3.0.tsv | sed 's/###/\t/g' | awk '{for (i=2; i<=NF; i+=1) {split($i,n,"="); print $1"\t"n[1]"\t"tolower(n[2])}}' | awk 'NR == FNR {keywords[$1]=1; next} {for (term in keywords) {if($2==term) {print; next}}}' metadata_field/included_keys_list.txt - | grep -wvf metadata_field/unavailable_terms_list.txt - | awk 'NF==3' > metadata_field/metadata.melted.tsv

### curate "Metadata" field
Rscript metadata_field/curate_metadata_field.R
### produce *CA_combined_informative_fields.tsv*


### extract metadata for each categorie
Rscript extract_metadata.freeze1.R CA_combined_informative_fields.tsv freeze1
### produce *metadata.freeze1.tsv* and *full_info_metadata.freeze1.tsv*

```
