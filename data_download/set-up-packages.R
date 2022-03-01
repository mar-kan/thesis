# install Bioconductor and check version
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()
BiocManager::version()

# install Bioconductor packages required
packages <- c("SummarizedExperiment", "TCGAbiolinks", "stringr", "MASS")

for (p in packages)
  BiocManager::install(p, dependencies=T)
