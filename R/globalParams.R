library("raster"); library("tidyverse"); library("sf"); library("lemon"); library("stringr")
library("tidyverse")

repo_root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."))
data_dir <- file.path(repo_root, "data")
tempFolder <- file.path(data_dir, "GOES")
dataFolder <- file.path(data_dir, "GOFER")

setwd(tempFolder)
fireData <- read.csv("fireData.csv",stringsAsFactors=F)
