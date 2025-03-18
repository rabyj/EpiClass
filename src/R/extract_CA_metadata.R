# Author: Frédérique White
# Date: 2025-03
# License: GPLv3

### Extract metadata from ChIP-Atlas

suppressMessages(library(data.table))
suppressMessages(library(dplyr))
suppressMessages(library(stringr))
suppressMessages(library(ggplot2))

match_term <- function(description, terms) {
	match <- terms[sapply(terms, function(term) grepl(term, description))]
	if (length(match) > 0) {
		return(match[1])  # Return the first matching term
  	} else {
		return(NA)  # Return NA if no match found
  }
}

### metadata
all_information_df = data.frame(fread("CA_metadata_4DB+all_pred.20240606.tsv", header=T))
metadata = all_information_df[, c("Experimental.id","Disease.state.name","Cell.type.description","Sex.geo",
	"Age","Tissue.type.ca","Tissue.type.cistromedb")]

colnames(metadata) = c("ID", "source_cancer_Disease.state.name","source_cancer_Cell.type.description",
	"source_sex_Sex.geo",
	"source_age_Age","source_age_Tissue.type.ca","source_age_Tissue.type.cistromedb"
	)

prediction = all_information_df[, c("Experimental.id","Predicted_class_cancer","Max_pred_cancer",
	"Predicted_class_sex","Max_pred_sex","Predicted_class_donorlife","Max_pred_donorlife")]
colnames(prediction)[1] = "ID"

rm(all_information_df)

#-------------------------------------------------------------------
cat(paste0("---> extracting cancer terms\n"))

cancer_keywords = unlist(read.csv("keywords_cancer.txt", header=F)$V1)
healthy_keywords = c("tissue_diagnosis=normal", "healthy", "resembles_a_tumor")

#metadata$source_cancer_Cell.type.description = ifelse(grepl("MeSH_", metadata$source_cancer_Cell.type.description), NA, metadata$source_cancer_Cell.type.description)
### extract cancer related keywords
metadata$extracted_cancer_dsn <- sapply(tolower(metadata$source_cancer_Disease.state.name), match_term, terms = c(cancer_keywords, healthy_keywords))
metadata$extracted_cancer_ctd <- sapply(tolower(metadata$source_cancer_Cell.type.description), match_term, terms = c(cancer_keywords, healthy_keywords))

metadata$extracted_cancer_dsn = ifelse(grepl("tissue_diagnosis=normal", metadata$extracted_cancer_ctd), NA, metadata$extracted_cancer_dsn)

metadata$extracted_cancer_dsn = ifelse(metadata$extracted_cancer_dsn %in% healthy_keywords, "non-cancer", metadata$extracted_cancer_dsn)
metadata$extracted_cancer_dsn = ifelse(metadata$extracted_cancer_dsn!="non-cancer" & !is.na(metadata$extracted_cancer_dsn), "cancer", metadata$extracted_cancer_dsn)
metadata$extracted_cancer_ctd = ifelse(metadata$extracted_cancer_ctd %in% healthy_keywords, "non-cancer", metadata$extracted_cancer_ctd)
metadata$extracted_cancer_ctd = ifelse(metadata$extracted_cancer_ctd!="non-cancer" & !is.na(metadata$extracted_cancer_ctd), "cancer", metadata$extracted_cancer_ctd)


### combine cancer-related fields
cancer_field_1 = na.omit(metadata[,c("ID", "extracted_cancer_dsn")])
cancer_field_2 = na.omit(metadata[,c("ID", "extracted_cancer_ctd")])
colnames(cancer_field_1) = colnames(cancer_field_2) = c("ID", "cancer")
all_cancer = data.frame(rbind(cancer_field_1, cancer_field_2))

### concat fields
full_result_cancer <- all_cancer %>%
 group_by(ID) %>%
 summarise(cancer = paste(sort(unique(cancer)), collapse = ":"))
full_result_cancer = data.frame(full_result_cancer)
nb_total = ncol(full_result_cancer)
full_result_cancer = full_result_cancer[grep(":", full_result_cancer$cancer, invert=T), ]
nb_clean = ncol(full_result_cancer)
cat(paste0(" * removing ",(nb_total - nb_clean), " inconsistent cancer metadata\n"))

metadata = merge(metadata, full_result_cancer, by="ID", all.x=T)

nb_metadata = nrow(subset(metadata, !is.na(cancer)))
cat(paste0(" * cancer extracted for ", nb_metadata, " samples\n"))

#-------------------------------------------------------------------
cat(paste0("---> extracting sex terms\n"))
metadata$sex <- sapply(tolower(metadata$source_sex_Sex.geo), match_term, terms = c("male_and_female","female","mix","pool","male"))
metadata$sex = ifelse(tolower(metadata$source_sex_Sex.geo)=="m", "male", metadata$sex)
metadata$sex = ifelse(tolower(metadata$source_sex_Sex.geo)=="f", "female", metadata$sex)
metadata$sex = gsub("mix|male_and_female|pool", "mixed", metadata$sex)

nb_metadata = nrow(subset(metadata, !is.na(sex)))
cat(paste0(" * sex extracted for ", nb_metadata, " samples\n"))

#-------------------------------------------------------------------
cat(paste0("---> extracting age terms using keywords\n"))
age_keywords = unlist(read.csv("keywords_life_stage.txt", header=F)$V1)

### clean source_age_Age field
metadata$source_age_Age = tolower(metadata$source_age_Age)
metadata$source_age_Age = ifelse(metadata$source_age_Age==0, NA, metadata$source_age_Age)
metadata$source_age_Age = ifelse(metadata$source_age_Age=="not_collected", NA, metadata$source_age_Age)
metadata$source_age_Age = ifelse(metadata$source_age_Age=="unknown", NA, metadata$source_age_Age)
metadata$source_age_Age = ifelse(grepl("passage", metadata$source_age_Age), NA, metadata$source_age_Age)
metadata$source_age_Age = ifelse(grepl("mixture", metadata$source_age_Age), NA, metadata$source_age_Age)
metadata$source_age_Age = ifelse(grepl("cultured", metadata$source_age_Age), NA, metadata$source_age_Age)

### extraction based on keywords
metadata$extracted_age <- sapply(metadata$source_age_Age, match_term, terms = age_keywords)
metadata$extracted_age_ca <- sapply(tolower(metadata$source_age_Tissue.type.ca), match_term, terms = age_keywords)
metadata$extracted_age_cistromedb <- sapply(tolower(metadata$source_age_Tissue.type.cistromedb), match_term, terms = age_keywords)

### combined extraction based on keywords
age_field_1 = na.omit(metadata[,c("ID", "extracted_age")])
age_field_2 = na.omit(metadata[,c("ID", "extracted_age_ca")])
age_field_3 = na.omit(metadata[,c("ID", "extracted_age_cistromedb")])
colnames(age_field_1) = colnames(age_field_2) = colnames(age_field_3) = c("ID", "age")
all_age_field = data.frame(rbind(age_field_1, age_field_2, age_field_3))

### uniform class names
all_age_field$age = gsub("fetus|fetal|neonat|cord_blood|umbilical|embryonic|juvenile|infant|gestation", "perinatal", all_age_field$age)
all_age_field$age = gsub("embryo", "perinatal", all_age_field$age)

full_result_age <- all_age_field %>%
 group_by(ID) %>%
 summarise(keyword_based_age = paste(sort(unique(age)), collapse = ":"))
full_result_age = data.frame(full_result_age)

metadata = merge(metadata, full_result_age, by="ID", all=T)

metadata$source_age_Age = ifelse(!is.na(metadata$keyword_based_age), NA, metadata$source_age_Age)

cat(paste0("---> extracting age terms using numbers\n"))
### extraction based on numerical information
metadata <- metadata %>%
mutate(
    # Extract numeric part
    age_value = as.numeric(str_extract(source_age_Age, "\\d+")),
    # Extract unit (year, month, etc.)
    age_unit = case_when(
      str_detect(source_age_Age, "day") ~ "day",
      str_detect(source_age_Age, "month") ~ "month",
      str_detect(source_age_Age, "week|wks") ~ "week",
      str_detect(source_age_Age, "year|yrs|yo|y") ~ "year",
      TRUE ~ NA_character_  # Fallback for unrecognized units
    )
  )
metadata = data.frame(metadata)

### convert numerical information in years from the reported unit
### we assumed that no reported unit == years
metadata$age_yo = metadata$age_value
metadata$age_yo = ifelse(metadata$age_unit=="month", metadata$age_value/12, metadata$age_yo)
metadata$age_yo = ifelse(metadata$age_unit =="week", metadata$age_value/52, metadata$age_yo)
metadata$age_yo = ifelse(metadata$age_unit =="day", metadata$age_value/365, metadata$age_yo)
metadata$age_yo = ifelse(is.na(metadata$age_unit), metadata$age_value, metadata$age_yo)

### convert numerical information to classes
metadata$numerical_based_age = NA
metadata$numerical_based_age = ifelse(metadata$age_yo <= 12, "child", metadata$numerical_based_age)
metadata$numerical_based_age = ifelse(metadata$age_yo <=2, "perinatal", metadata$numerical_based_age)
metadata$numerical_based_age = ifelse(metadata$age_yo > 12, "adult", metadata$numerical_based_age)

### combine age-related fields
age_field_1 = na.omit(metadata[,c("ID", "keyword_based_age")])
age_field_2 = na.omit(metadata[,c("ID", "numerical_based_age")])
colnames(age_field_1) = colnames(age_field_2) = c("ID", "age")

all_age_field = data.frame(rbind(age_field_1, age_field_2))
full_result_age <- all_age_field %>%
 group_by(ID) %>%
 summarise(donorlife = paste(sort(unique(age)), collapse = ":"))
full_result_age = data.frame(full_result_age)

metadata = merge(metadata, full_result_age, by="ID", all=T)

nb_metadata = nrow(subset(metadata, !is.na(donorlife)))
cat(paste0(" * life-stage extracted for ", nb_metadata, " samples\n"))

### add predictions
prediction$Predicted_class_donorlife = gsub("embryonic", "perinatal", prediction$Predicted_class_donorlife)
prediction$Predicted_class_donorlife = gsub("fetal", "perinatal", prediction$Predicted_class_donorlife)
prediction$Predicted_class_donorlife = gsub("newborn", "perinatal", prediction$Predicted_class_donorlife)

full_metadata = metadata
full_metadata = merge(full_metadata, prediction, by="ID", all=T)
write.table(full_metadata, file="full_info_extracted_metadata.tsv", sep="\t",row.names=F, quote=F)

metadata = metadata[, c("ID", "cancer", "sex", "donorlife")]
metadata = merge(metadata, prediction, by="ID", all=T)
output_metadata = metadata
colnames(output_metadata)[2] = "expected_cancer"
colnames(output_metadata)[3] = "expected_sex"
colnames(output_metadata)[4] = "expected_donorlife"
write.table(output_metadata, file="extracted_metadata.tsv", sep="\t",row.names=F, quote=F)


### report from comparing with prediction
get_report <- function(df, category){

	predicted_field = paste0("Predicted_class_", category)

	avail = nrow(subset(df, !is.na(df[,category])))
	missing = nrow(subset(df, is.na(df[,category])))
	matching = nrow(subset(df, df[,category]==df[,predicted_field]))
	mismatching = nrow(subset(df, df[,category]!=df[,predicted_field]))
	return(list(avail=avail, missing=missing, matching=matching, mismatching=mismatching))
}

report = data.frame(matrix(data=NA, nrow=9, ncol=7))
colnames(report) = c("category", "available", "missing", "match", "mismatch", "rate", "total")
i = 1
for (category in c("cancer", "sex", "donorlife")){

	score = paste0("Max_pred_", category)
	high_conf = subset(metadata, metadata[, score]>=0.6)
	low_conf = subset(metadata, metadata[, score]<0.6)

	# total
	rep = get_report(metadata, category)
	report[i, ] = c(category, rep$avail, rep$missing, rep$matching, rep$mismatching, round(100*rep$matching/rep$avail,2), nrow(metadata))
	i = i + 1

	# High conf
	rep = get_report(high_conf, category)
	report[i, ] = c(paste0(category,"_0.6"), rep$avail, rep$missing, rep$matching, rep$mismatching, round(100*rep$matching/rep$avail,2), nrow(high_conf))
	i = i + 1

	# Low conf
	rep = get_report(low_conf, category)
	report[i, ] = c(paste0(category,"_lowconf"), rep$avail, rep$missing, rep$matching, rep$mismatching, round(100*rep$matching/rep$avail,2), nrow(low_conf))
	i = i + 1
}
write.table(report, file="report.tsv", sep="\t",row.names=F, quote=F)


### create donut charts from report
pdf("donut_charts.pdf", width=3, height=4)
donut_colors = c("match"="#00a759", "mismatch"="#f12828", "missing"="#e6e6e6",
		"lowconf"="#deaa87","highconf"="#00a3a3","avail"="#ffd42a")

for (category in c("cancer", "sex", "donorlife")){

	string = paste0(category,"_0.6")
	highconf = subset(report, category==string)

	string = paste0(category,"_lowconf")
	lowconf = subset(report, category == string)

	df = data.frame(donut1 = c("match", "mismatch", "lowconf","missing","missing"),
				donut2 = c("highconf", "highconf", "lowconf","lowconf", "highconf"),
				donut3 = c("avail", "avail", "avail", "missing", "missing"),
				min = c(0, highconf$match, highconf$mismatch, lowconf$available,lowconf$missing),
				max = c(highconf$match, highconf$mismatch, lowconf$available, lowconf$missing, highconf$missing)
		)
	df$ymin = cumsum(df$min)
	df$ymax = cumsum(df$max)
	chart = ggplot(df) +
		geom_rect(aes(fill=donut1, ymax=ymax, ymin=ymin, xmax=3.5, xmin=2.5)) +
		geom_rect(aes(fill=donut2, ymax=ymax, ymin=ymin, xmax=2, xmin=1)) +
		geom_rect(aes(fill=donut3, ymax=ymax, ymin=ymin, xmax=2.75, xmin=2.5)) +
		xlim(c(0, 4)) + theme_void()+coord_polar(theta="y")+
		scale_fill_manual(values = donut_colors, guide = "none") + ggtitle(category)
	print(chart)
}
dev.off()
