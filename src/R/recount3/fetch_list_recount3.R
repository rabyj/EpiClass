

# Rscript fetch_list_recount3.R organism

# ml r/4.4.0

#if (!require("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#
#BiocManager::install("recount3")

suppressMessages(library("recount3"))

args = commandArgs(trailingOnly=TRUE)
organism = args[1] # human or mouse

projects <- available_projects(organism=organism)
write.table(projects, file=paste0(organism,"/available_projects.tsv"), row.names=F, quote=F)

samples <- available_samples(organism=organism)
write.table(samples, file=paste0(organism,"/available_samples.tsv"), row.names=F, quote=F)

# <recount3_url>/<organism>/data_sources/<data_source>/base_sums/<last 2 project letters or digits>/<project>/<last 2 sample letters or digits>/<data_source>.base_sums.<project>_<sample>.ALL.bw
