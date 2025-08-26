suppressMessages(library(data.table))
suppressMessages(library(tidyr))
suppressMessages(library(stringr))
suppressMessages(library(dplyr))

metadata_df = data.frame(fread("CA_metadata_4DB+all_pred.20240606_mod3.0.tsv", header=T, quote=""))[, c(1,2)]
colnames(metadata_df) = c("ID","tmp")
#for (column in colnames(metadata_df)[-1]){
#     metadata_df[,column] = tolower(gsub("-", "_", metadata_df[,column]))
#     metadata_df[,column] = ifelse(metadata_df[,column]=="_" | metadata_df[,column]=="____", NA, metadata_df[,column])
#}

df = read.csv("metadata_field/metadata.melted.tsv", sep="\t", header=F)
colnames(df) = c("ID", "key", "value")
df$value = gsub("-", "_", df$value)

df_wide <- pivot_wider(df, names_from = key, values_from = value, values_fill = NA, values_fn = function(x) paste0(x, collapse = "_"))
### Cleaning and aggregating metadata fields

df_wide = merge(df_wide, metadata_df, by="ID", all=T)
df_wide$tmp=NULL

#--------------------------------------
### 1- sex-related cleaning

# "Sex" key contains "male", "female", "mixed", "male_and_female" or unwanted
df_wide$Sex = ifelse(str_detect(df_wide$Sex, "and|pooled"), NA, df_wide$Sex)

# "cell_sex" and "sex-geo" keys contain "m", "f", "b" or "u"
df_wide$cell_sex = gsub("m", "male", df_wide$cell_sex)
df_wide$cell_sex = gsub("f", "female", df_wide$cell_sex)
df_wide$cell_sex = ifelse(str_detect(df_wide$cell_sex, "female|male"), df_wide$cell_sex, NA)

df_wide = unite(df_wide, col=combined_sex, c(Sex, sex, cell_sex), sep='_', remove=T, na.rm=T)

#--------------------------------------
### 2- cancer-related cleaning
df_wide$cancer_source_name = gsub("luekemia","leukemia", df_wide$source_name)
df_wide$cancer_source_name = ifelse(str_detect(df_wide$cancer_source_name, "aml"), "acute_myeloid_leukemia", df_wide$cancer_source_name)
df_wide$cancer_source_name = ifelse(str_detect(df_wide$cancer_source_name, "t_all|b_all"), "acute_lymphoblastic_leukemia", df_wide$cancer_source_name)
df_wide$cancer_source_name = ifelse(str_detect(df_wide$cancer_source_name, "cml"), "chronic_myelogenous_leukemia", df_wide$cancer_source_name)
df_wide$cancer_source_name = ifelse(str_detect(df_wide$cancer_source_name, "cll"), "chronic_lymphocytic_leukemia", df_wide$cancer_source_name)


df_wide$cell_type = ifelse(str_detect(df_wide$cell_type, "aml"), "acute_myeloid_leukemia", df_wide$cell_type)
df_wide$cell_type = ifelse(str_detect(df_wide$cell_type, "t_all|b_all"), "acute_lymphoblastic_leukemia", df_wide$cell_type)
#df_wide$cell_type = ifelse(df_wide$cell_type=="all", "acute_lymphoblastic_leukemia", df_wide$cell_type)
df_wide$cell_type = ifelse(str_detect(df_wide$cell_type, "cml"), "chronic_myelogenous_leukemia", df_wide$cell_type)
df_wide$cell_type = ifelse(str_detect(df_wide$cell_type, "cll"), "chronic_lymphocytic_leukemia", df_wide$cell_type)

df_wide$disease_state = ifelse(df_wide$disease_state=="hcc", "hepatocellular_carcinoma", df_wide$disease_state)
df_wide$disease_state = ifelse(df_wide$disease_state=="ad", "alzheimer", df_wide$disease_state)
df_wide$disease_state = ifelse(str_detect(df_wide$disease_state, "cll"), "chronic_lymphocytic_leukemia", df_wide$disease_state)
df_wide$disease_state = ifelse(str_detect(df_wide$disease_state, "t_all"), "acute_lymphoblastic_leukemia", df_wide$disease_state)
df_wide$disease_state = ifelse(str_detect(df_wide$disease_state, "aml"), "acute_myeloid_leukemia", df_wide$disease_state)

df_wide$disease = ifelse(str_detect(df_wide$disease, "cml"), "chronic_myelogenous_leukemia", df_wide$disease)
df_wide$disease = ifelse(str_detect(df_wide$disease, "aml"), "acute_myeloid_leukemia", df_wide$disease)
df_wide$disease = ifelse(str_detect(df_wide$disease, "t_all"), "acute_lymphoblastic_leukemia", df_wide$disease)

df_wide$diagnosis = ifelse(str_detect(df_wide$diagnosis, "cll"), "chronic_lymphocytic_leukemia", df_wide$diagnosis)
df_wide = unite(df_wide, col=combined_disease, c(disease, disease_state, diagnosis), sep='_', remove=T, na.rm=T)

#--------------------------------------
### 3- lifestage-related cleaning

# "age" key contains values such as "63_years"
# Numerical values and their units are extracted then classified into group "adult", "child" and "perinatal".
# wpc and pcw = week post conception
age_df <- df_wide[, c("ID", "age")] %>%
mutate(age_value = as.numeric(str_extract(age, "\\d+\\.?\\d*")),
		age_unit = as.numeric(case_when(str_detect(age, "day") ~ "365",str_detect(age, "month") ~ "12",
			str_detect(age, "week|wk|pcw|wpc|w.p.c.") ~ "52", str_detect(age, "year|yr|yo") ~ "1", TRUE ~ NA_character_ )))

age_df = as.data.frame(subset(age_df, age_value!=0))

# devide numerical value by units to get age in years
age_df$age_year = ifelse(!is.na(age_df$age_unit), age_df$age_value/age_df$age_unit, NA)

# when no unit extracted, use as is if the numerical value fit the entire field or contain "y"
age_df$age_year = ifelse((age_df$age == age_df$age_value | age_df$age == paste0(age_df$age_value,"y")) | age_df$age == paste0(age_df$age_value,"_y"), age_df$age_value, age_df$age_year)

# classified into group
age_df$age_group = ifelse(age_df$age_year < 12, "child", "adult")
age_df$age_group = ifelse(age_df$age_year <1, "perinatal", age_df$age_group)

df_wide = merge(df_wide, age_df[, c("ID","age_group")], by="ID", all.x=T)

# aggregate lifestage-related fields
df_wide = unite(df_wide, col=combined_dev_stage, c(dev_stage, developmental_stage), sep='_', remove=T, na.rm=T)

#--------------------------------------
### 4- biomaterial-related cleaning

cell_line_list = c("cell_line","lncap","wa01", "hela","t47d","u2os","mcf_7","mcf7","wi_38","caco_2",
	"hek_293","hek293","mcf10a","hudep2","rh4","mda_mb_231","ist_mes2","k562","thp_1","thp1","wibr3","molm_13","a673","ishikawa",
	"hct116","hepg2","imr90","hap1","jurkat","a549", "gm12878", "gm19193","gm19099","gm18951","gm18526","gm18505","gm15510","gm12892","gm12891","gm10847","gm20000","gm13977","gm13976","gm12874","gm12873","gm12872","gm12866","gm12801","gm10266","gm10248","gm08714")

# "cell_line" field contains cell line name and cell information, default to "cell_line"
df_wide$cell_line = ifelse(str_detect(df_wide$cell_line, paste0(cell_line_list,collapse="|")), "cell_line", df_wide$cell_line)

# "sample_type" field contains biomaterial information
df_wide$sample_type = ifelse(df_wide$sample_type=="tissue", "primary_tissue", df_wide$sample_type)
# aggregate biomaterial-related fields
df_wide = unite(df_wide, col=combined_sample_type, c(sample_type, biosample_type), sep='_', remove=T, na.rm=T)

# "source_name" field contains biomaterial information and other
df_wide$biomat_source_name = ifelse(str_detect(df_wide$source_name, paste0(cell_line_list,collapse="|")), "cell_line", df_wide$source_name)

# "strain" field contains cell line name and cell information, default to "cell_line"
df_wide$strain = ifelse(str_detect(df_wide$strain, paste0(cell_line_list, collapse="|")), "cell_line", NA)

# "cell" field contains cell line name or cell information, default to "cell_line"
df_wide$cell = ifelse(str_detect(df_wide$cell, paste0(cell_line_list, collapse="|")), "cell_line", NA)

# aggregate biomaterial-related fields
df_wide = unite(df_wide, col=combined_strain, c(cell, strain), sep='_', remove=T, na.rm=T)


df_wide[df_wide==""] = NA
write.table(df_wide, file="CA_combined_informative_fields.tsv", sep="\t", row.names=F, quote=F)



#--------------------------------------
### 5- biospecimen-related cleaning

# aggregate bispecimen-related fields
#df_wide = unite(df_wide, col=cell_type, c(cell_type, cell_types), sep='_', remove=T, na.rm=T)
