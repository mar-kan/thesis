# changes current directory if it is not the right one
library(stringr)
if (!str_detect(getwd(), "data_download"))
  setwd("./data_download")

source("functions.R")
loadLibraries()

# list of 33 projects to be downloaded
projects <- c("ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH",
              "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG",
              "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM")

# matrix that stores the dimensions of the datasets
dims <- matrix(, nrow=33, ncol=2)
row.names(dims) <- projects

# downloads each tumor type and saves it to a csv file
for (i in 1:length(projects))
{
  data <- downloadData(projects[i])
  data_assay <- assay(data, "raw_count") # stores RNA-sequences counts

  # add labels to datasets
  labels <- x6<-rep(projects[i], ncol(data_assay))
  data_assay <- rbind(data_assay, labels)

  # write data to csv file
  filename <- paste0(projects[i], ".csv")
  write.csv(data_assay, file=paste0('../All_Datasets/', filename), row.names=T)

  # write their dimensions
  dims[i, 1] <- nrow(data_assay)
  dims[i, 2] <- ncol(data_assay)
}

# write all dimensions in csv file
write.matrix(dims, file= "../Dataset/dims.csv", sep=",")
