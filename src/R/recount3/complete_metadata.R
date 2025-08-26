
args = commandArgs(trailingOnly=TRUE)
organism = args[1]

project_df = read.csv(paste0(organism,"/available_projects.tsv"), sep="\t")
sample_df = read.csv(paste0(organism,"/available_samples.tsv"), sep="\t")

if (organism == "human"){
	missing_df = read.csv("missing_samples.txt", sep="\t", header=F)
	colnames(missing_df) = c("external_id","project","project_suffix","BigWigURL")
	missing_study_list = unique(missing_df$project)
}

for (project_id in missing_study_list){

	sra_df = read.csv(paste0("~/scratch/recount3_",organism,"/missing/sra.sra.",project_id,".MD.gz"), sep="\t")
	colnames(sra_df) = c(colnames(sra_df)[1:3], paste0("sra.",colnames(sra_df)[4:ncol(sra_df)]))
	colnames(sra_df)[4] = paste0(colnames(sra_df)[4], ".x")

	recountproject_df = subset(sample_df, project == project_id)
	colnames(recountproject_df) = c(colnames(recountproject_df)[1], paste0("recount_project.",colnames(recountproject_df)[-1]))
	colnames(recountproject_df)[6] = "recount_project.metadata_source"
	recountproject_df = recountproject_df[, c("external_id", "recount_project.project", "recount_project.organism", "recount_project.file_source","recount_project.metadata_source","recount_project.date_processed")]

	combined_df = merge(sra_df, recountproject_df, by="external_id", all.x=T)

	recountqc_df = read.csv(paste0("~/scratch/recount3_",organism,"/missing/sra.recount_qc.",project_id,".MD.gz"), sep="\t")[, -c(2:3)]
	colnames(recountqc_df) = c(colnames(recountqc_df)[1], paste0("recount_qc.",colnames(recountqc_df)[2:ncol(recountqc_df)]))

	combined_df = merge(combined_df, recountqc_df, by="rail_id", all.x=T)


	recountseqqc_df = read.csv(paste0("~/scratch/recount3_",organism,"/missing/sra.recount_seq_qc.",project_id,".MD.gz"), sep="\t")[, -c(2:3)]
	colnames(recountseqqc_df) = sub("X","", colnames(recountseqqc_df))
	colnames(recountseqqc_df) = tolower(colnames(recountseqqc_df))
	colnames(recountseqqc_df) = c(colnames(recountseqqc_df)[1], paste0("recount_seq_qc.",colnames(recountseqqc_df)[2:ncol(recountseqqc_df)]))

	combined_df = merge(combined_df, recountseqqc_df, by="rail_id", all.x=T)


	combined_df$recount_pred.sample_acc.y = NA
	combined_df$recount_pred.curated.type = NA
	combined_df$recount_pred.curated.tissue = NA
	combined_df$recount_pred.pattern.predict.type = NA
	combined_df$recount_pred.pred.type = NA
	combined_df$recount_pred.curated.cell_type = NA
	combined_df$recount_pred.curated.cell_line = NA

	combined_df = merge(combined_df, missing_df[, -1], by="external_id")
	combined_df = cbind(rail_id=combined_df$rail_id, combined_df[,-2])

	n_missing = nrow(subset(missing_df, project == project_id))
	n_added = nrow(combined_df)
	if ( ncol(combined_df) == 175 & n_missing == n_added){
		print(paste0(" * project ", project_id, " is done "))
		write.table(combined_df, file=paste0("~/scratch/recount3_",organism,"/missing_formated/recount3_",organism,"_",project_id,"_sample_metadata.tsv"), sep="\t", row.names=F, quote=F)
	} else {
		print(paste0(" * project ", project_id, " is NOT okay ! "))
	}

}
