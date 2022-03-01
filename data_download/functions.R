loadLibraries <- function()
{
  library("TCGAbiolinks")
  library("MASS")
  library("SummarizedExperiment")
}


downloadData <- function(project)
{
  query <- GDCquery(project = paste0("TCGA-", project),
                   data.category = "Gene expression",
                   data.type="Gene expression quantification",
                   experimental.strategy = "RNA-Seq",
                   platform = "Illumina HiSeq",
                   file.type = "results",
                   sample.type = c("Primary Tumor","Primary Blood Derived Cancer - Peripheral Blood",
                                   "Primary Blood Derived Cancer - Bone Marrow"),
                   legacy = TRUE
  )

  GDCdownload(query)
  return (GDCprepare(query = query))
}