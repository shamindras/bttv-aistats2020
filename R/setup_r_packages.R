#!/usr/bin/env Rscript

# Install required packages from CRAN and devtools
req_pckgs <- c("devtools", "tidyverse", "glue",
               "here", "janitor")
install.packages(pkgs = req_pckgs, dependencies = TRUE,
                 repos = "http://cran.us.r-project.org")
devtools::install_github(repo = "ryurko/nflscrapR")
devtools::install_github("ryurko/nflWAR")
