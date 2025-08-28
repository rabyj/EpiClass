
#====================================================================
# Extraction of ENCODE metadata
#====================================================================

suppressMessages(library(data.table))
suppressMessages(library(dplyr))
suppressMessages(library(stringr))
suppressMessages(library(ggplot2))

args = commandArgs(trailingOnly=TRUE)
version = args[1]


### metadata fields used for each category
field_list = list()

field_list[["cancer"]] = c("BIOSAMPLE_health_status",
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

all_field_list = unique(unlist(field_list))

metadata_df = data.frame(fread("encode_combined_informative_fields.tsv", header=T))[, c("ID", all_field_list)]

extract_metadata <- function(metadata_df, field_list, keyword_df, category) {

	# stack all fields
	evaluated_fields = list()
	for (field in field_list) {
  		temp_field <- metadata_df[, c("ID", field)]
		colnames(temp_field) <- c("ID", "metadata")
		evaluated_fields[[field]] <- temp_field
	}
	combined_fields <- do.call(rbind, evaluated_fields)
	combined_fields = na.omit(subset(combined_fields, metadata!=""))

	# search for keywords
	combined_fields$match <- sapply(combined_fields$metadata, match_term, terms = keyword_df$keyword)

	# map keywords to class
	df = merge(combined_fields, keyword_df, by.x="match", by.y="keyword")

	concatenated_df <- df %>%
	 group_by(ID) %>%
	 summarise(all_match = paste(sort(unique(match)), collapse = ":"),
	 	clean = paste(sort(unique(class)), collapse = ":"))
	concatenated_df = data.frame(concatenated_df)

	# assign indeterminate to inconsistencies or misleading keywords (such as tumor_associated)
	concatenated_df$clean = ifelse(str_detect(concatenated_df$clean, ":"), "indeterminate",concatenated_df$clean)
	concatenated_df$clean = gsub("ND", "indeterminate", concatenated_df$clean)

	category_research_field = paste0("extracted_terms_", category)
	category_field = paste0("expected_", category)
	colnames(concatenated_df) = c("ID", category_research_field, category_field)

	return(concatenated_df)
}

match_term <- function(description, terms) {
	match <- terms[sapply(terms, function(term) grepl(term, description))]
	if (length(match) > 0) {
		return(match[1])  # Return the first matching term
  	} else {
		return(NA)  # Return NA if no match found
  }
}


#====================================================================
# Metadata extraction
#====================================================================
category_list = names(field_list)

for (category in category_list){
	cat(paste0("extracting ",category," ...\n"))
	keywords_df = read.csv(paste0("keywords_",category,".tsv"), sep="\t", header=T)

	category_df = extract_metadata(metadata_df, field_list[[category]], keywords_df, category)
	metadata_df = merge(metadata_df, category_df, by="ID", all.x=T)

}
metadata_df = metadata_df[, c("ID","extracted_terms_cancer","expected_cancer")]
metadata_df[metadata_df==""] = NA
write.table(metadata_df, file=paste0("extracted_metadata.",version,".tsv"), sep="\t", row.names=F, quote=F)

#previous_version = read.csv("encode_metadata_2023-10-25_clean-v2.cancer_status.tsv", sep="\t")
#df = merge(previous_version, metadata_df[,c("ID","expected_cancer")], by.x="md5sum", by.y="ID")
#write.table(df, file=paste0("comparison.",version,".tsv"), sep="\t", row.names=F, quote=F)
