#!/usr/bin/env Rscript

# Here we simply download the main data sources provided by Prof. John Hunter
# (PSU) as found here: http://personal.psu.edu/drh20/code/btmatlab/

library(tidyverse)

get_nascar_2002_dat <- function(){
    fname_pfx <- "nascar2002"
    fname_ext <- c(".mat", ".txt", ".xls")
    fnames <- glue::glue("{fname_pfx}{fname_ext}")
    base_url <- "http://personal.psu.edu/drh20/code/btmatlab/"
    furls <- glue::glue("{base_url}{fnames}")
    fpaths <- glue::glue("{here::here('data', 'nascar')}/{fnames}")

    # Download all datasets to the specified nascar folder
    purrr::walk2(.x = furls, .y = fpaths,
                 .f = ~utils::download.file(url = .x, destfile = .y))
}

# Download the (raw) NASCAR 2002 data
print("Getting NASCAR 2002 datasets...")
get_nascar_2002_dat()
print(glue::glue("\nDONE!\nPlease see {here::here('data', 'nascar')} for the raw NASCAR 2002 datasets"))