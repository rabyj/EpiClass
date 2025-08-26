
#====================================================================
# Extraction of ChIP-Atlas metadata
#====================================================================

suppressMessages(library(data.table))
suppressMessages(library(dplyr))
suppressMessages(library(stringr))
suppressMessages(library(ggplot2))

args = commandArgs(trailingOnly=TRUE)
input_file = args[1] # "CA_combined_informative_fields.tsv"
version = args[2]


### metadata fields used for each category
field_list = list()

field_list[["sex"]] =  c("combined_sex")
field_list[["biomat"]] = c("biomat_source_name" ,"combined_sample_type","combined_strain", "cell_line")
field_list[["lifestage"]] =  c("age", "age_group", "combined_dev_stage", "cell_description","source_name","cell_type","tissue")
field_list[["cancer"]] = c("cancer_source_name","cell_type","tissue","combined_disease","cell_description")
field_list[["biospecimen"]] = c("source_name" ,"cell_type","tissue","cell_description","cell_lineage")

all_field_list = unique(unlist(field_list))

metadata_df = data.frame(fread(input_file, header=T))[, c("ID", all_field_list)]

prediction_df = read.csv("EpiClass_predictions_20240606_mod3.tsv", sep="\t")
colnames(prediction_df)[1] = "ID"
prediction_df$predicted_lifestage = gsub("embryonic|fetal|newborn", "perinatal", prediction_df$predicted_lifestage)



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

	# compare with EpiClass predictions
	if ( category != "biospecimen"){
		predicted_field = paste0("predicted_", category)
		score = paste0("score_", category)
		df = merge(concatenated_df, prediction_df, by="ID")
		#total
		avail = nrow(subset(df, !is.na(df[, category_field]) & df[, category_field]!= "indeterminate" ))
		matching = nrow(subset(df, df[, category_field]==df[,predicted_field]))
		cat(paste0(" --> All ",category," fields (total): ",avail, " extracted (" ,round(100*matching/avail,2), "%)\n"))
	}

	return(concatenated_df)
}

combine_fields <- function(df, field_list) {
	all_fields = list()
	for (field in field_list) {

		temp_field <- df[, c("ID", field)]
		colnames(temp_field) <- c("ID", "metadata")

		temp_field$metadata = gsub("-|0", "", temp_field$metadata)
		temp_field$metadata = tolower(temp_field$metadata)
		all_fields[[field]] <- temp_field
	}
	combined_fields <- do.call(rbind, all_fields)
	combined_fields = subset(combined_fields, metadata!="")
	return(combined_fields)
}

match_term <- function(description, terms) {
	match <- terms[sapply(terms, function(term) grepl(term, description))]
	if (length(match) > 0) {
		return(match[1])  # Return the first matching term
  	} else {
		return(NA)  # Return NA if no match found
  }
}

concatenate_fields <- function(df, term_dict) {
	df = merge(df, term_dict, by.x="terms", by.y="keyword")
	concatenated_df <- df %>%
	 group_by(ID) %>%
	 summarise(clean = paste(sort(unique(class)), collapse = ":"))
	concatenated_df = data.frame(concatenated_df)

	nb_total = nrow(concatenated_df)
	concatenated_df = concatenated_df[grep(":", concatenated_df$clean, invert=T), ]
	nb_clean = nrow(concatenated_df)
	cat(paste0(" * removing ",(nb_total - nb_clean), " inconsistent metadata\n"))
	return(concatenated_df)
}

evaluate_field <- function(metadata_df, field, keyword_df, category){
	metadata_df$expected = sapply(metadata_df[, field], match_term, terms = keyword_df$keyword)
	df = merge(metadata_df, keyword_df, by.x="expected", by.y="keyword")

	predicted_field = paste0("predicted_", category)
	score = paste0("score_", category)
	df = merge(df, prediction_df, by="ID")
	#df = subset(df, df[, score] > 0.6)

	avail = nrow(subset(df, !is.na(df$class) & df$class != "indeterminate"))
	matching = nrow(subset(df, df$class==df[,predicted_field]))
	cat(paste0(field, ": ", matching," / ", avail, " extracted (" ,round(100*matching/avail,2), "%)\n"))
}


#====================================================================
# Metadata extraction
#====================================================================
category_list = names(field_list)

for (category in category_list){
	cat(paste0("extracting ",category," ...\n"))
	keywords_df = read.csv(paste0("keywords_",category,".tsv"), sep="\t", header=T)

	if ( category != "biospecimen"){
		for (field in field_list[[category]]){
			evaluate_field(metadata_df, field, keywords_df, category)
		}
	}
	category_df = extract_metadata(metadata_df, field_list[[category]], keywords_df, category)
	metadata_df = merge(metadata_df, category_df, by="ID", all.x=T)

}
write.table(merge(metadata_df, prediction_df, by="ID", all.x=T), file=paste0("full_info_metadata.",version,".tsv"), sep="\t", row.names=F, quote=F)

metadata_df = metadata_df[, c("ID", grep("expected", colnames(metadata_df), value=T))]
metadata_df = merge(metadata_df, prediction_df, by="ID", all.x=T)
metadata_df[metadata_df==""] = NA
write.table(metadata_df, file=paste0("metadata.", version,".tsv"), sep="\t", row.names=F, quote=F)
