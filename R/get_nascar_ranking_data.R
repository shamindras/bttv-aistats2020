#!/usr/bin/env Rscript

# Here we download the main NASCAR 2002 ranking dataset for all 36 races as
# found here: https://fantasyracingcheatsheet.com/nascar/drivers/point-standings/2002
# We do minor modifications to the column names and add columns for year (2002)
# and the specific race number

library(glue)
library(here)
library(tidyverse)

#' Get NASCAR race ranking based on specific year and race number. All of the
#' data is sourced from \url{https://fantasyracingcheatsheet.com/nascar/drivers/point-standings}
#'
#' @param nsc_year : (integer) : The NASCAR year e.g. 2002
#' @param nsc_race : (integer) : The NASCAR race number, for 2002 this would
#' range from 1-36
#'
#' @export
get_nascar_standings <- function(nsc_year, nsc_race){

    nsc_url <- glue::glue("https://fantasyracingcheatsheet.com/nascar/drivers/",
                          "point-standings/{nsc_year}/{nsc_race}")

    nsc_html <- xml2::read_html(x = nsc_url)

    out_tbl <- nsc_html %>%
        rvest::html_nodes("table") %>%
        .[1] %>%
        rvest::html_table(fill = TRUE) %>%
        purrr::pluck(.x = ., 1) %>%
        dplyr::mutate(.data = .,
                      year = nsc_year,
                      race_num = nsc_race) %>%
        dplyr::rename(.data = .,
                      "pls_min_mv" = "+/-",
                      "top_5" = "Top-5",
                      "top_10" = "Top-10")

    base::return(out_tbl)
}

# Define GLOBAL variables ------------------------------------------------------

NASCAR_YR <- 2002
RACE_NUMS <- 1:36

# Obtain a combined data frame of nascar rankings for all races in a single tibble
print("Downloading the NASCAR 2002 rankings dataset...")
nascar_2002_all_std <-
    RACE_NUMS %>%
    purrr::map_dfr(.x = .,
                   .f = ~get_nascar_standings(nsc_year = NASCAR_YR,
                                              nsc_race = .x)) %>%
    janitor::clean_names()
print("Downloading DONE!")

# Download the (raw) NASCAR 2002 rankings data
# Write output to CSV format
print("Writing the NASCAR 2002 rankings dataset...")

readr::write_csv(x = nascar_2002_all_std,
                 path = here::here("data", "nascar", "nascar_2002_all_std.csv"))

print(glue::glue("DONE!\nPlease see {here::here('data', 'nascar', 'nascar_2002_all_std.csv')} for the NASCAR 2002 ranking csv file"))