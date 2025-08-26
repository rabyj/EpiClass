
suppressMessages(library(tidyr))
suppressMessages(library(stringr))
suppressMessages(library(dplyr))

field_list = c("BIOSAMPLE_health_status",
	"BIOSAMPLE_disease_term_name",
	"BIOSAMPLE_description",
	"BIOSAMPLE_TYPE_cell_slims",
	"BIOSAMPLE_TYPE_synonyms",
	"BIOSAMPLE_TYPE_organ_slims",
	"BIOSAMPLE_summary",
	"BIOSAMPLE_simple_summary",
	"EXPERIMENT_description",
	"EXPERIMENT_biosample_summary",
	"EXPERIMENT_simple_biosample_summary",
	"EXPERIMENT_internal_tags",
	"BIOSAMPLE_TYPE_system_slims",
	"BIOSAMPLE_TYPE_aliases",
	"BIOSAMPLE_TYPE_term_name")

df = read.csv("encode_full_metadata_2025-02_no_revoked.tsv", sep="\t", header=T)
df[df==""]=NA
colnames(df)[3] = "ID"
df = df[, c("ID", field_list)]

### Cleaning and aggregating metadata fields
#--------------------------------------
for (field in field_list){
	df[, field] = tolower(df[,field])
	df[, field] = gsub("unknown", NA, df[, field])
	df[, field] = gsub("\\[|\\]|'", "", df[, field])
	df[, field] = gsub("cancer_institute", "", df[, field])
	df[, field] = gsub("-", "_", df[, field])
	df[, field] = gsub("t_all", "acute_lymphoblastic_leukemia", df[, field])
}

df[df==""] = NA
write.table(df, file="encode_combined_informative_fields.tsv", sep="\t", row.names=F, quote=F)
