
library(dplyr)

args = commandArgs(trailingOnly=TRUE)
organism = args[1]
list_of_fields = args[2]
list_of_incomplete = args[3]

field_list = unlist(read.csv(list_of_fields, header=F)$V1)
file_paths = unlist(read.csv(list_of_incomplete, header=F)$V1)

cat("open all files ...\n")
read_and_clean <- function(file) {
	df <- read.csv(file, sep = "\t", colClasses = "character", stringsAsFactors = FALSE)  # Read tab-separated file
	if(! "sra.sample_acc.x" %in% colnames(df)){
		colnames(df) = sub("sra.sample_acc","sra.sample_acc.x",colnames(df))
	}
	return(df)
}
data_list <- lapply(file_paths, read_and_clean)


cat("merge all files ...\n")
add_missing_field <- function(df) {
	df[setdiff(field_list, colnames(df))] <- NA  # Add missing columns
	df <- df[field_list]  # Reorder columns to match the final structure
	return(df)
}
standardized_data <- lapply(data_list, add_missing_field)

merged_data <- bind_rows(standardized_data)

merged_data[] <- lapply(merged_data, function(column) {
	column <- gsub(" ", "_", column)       # Replace spaces with underscores
	column[column == ""] <- NA            # Replace empty strings with NA
	return(column)
	})

write.table(merged_data, file="human/merged_metadata_part2.tsv", sep="\t",row.names=F, quote=F)
