#!/usr/bin/env Rscript

# The following script will give us the pairwise Bradley-Terry
# model data for NFL

#-------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------
# File for building WP model

library(tidyverse)
library(here)
devtools::install_github("ryurko/nflWAR")
library(nflWAR)

#-------------------------------------------------------------------
# Define Variables
#-------------------------------------------------------------------

all_seasons <- 2009:2016
nfl_csv_gh_ref <- "https://raw.github.com/ryurko/nflscrapR-data/master/legacy_data/season_games/games_"

#-------------------------------------------------------------------
# Get all win-loss data for all seasons
# Source: https://github.com/ryurko/nflscrapR-models/blob/master/R/init_models/init_wp_model.R#L25-L28
#-------------------------------------------------------------------

# Load and stack the games data for all specified seasons
games_data <-  all_seasons %>%
    purrr::map_dfr(.,
                   function(x) {
                       suppressMessages(
                           readr::read_csv(paste(nfl_csv_gh_ref,
                                                 x, ".csv", sep = "")))
                   })

#-------------------------------------------------------------------
# Basic checks on Data
#-------------------------------------------------------------------

# Just check the colnames for reference
base::colnames(games_data)

# We expect 256 rows of data per season
# - 32 teams playing 16 games each
# Note: Indeed this is the case!
# TODO: write a test
games_data %>%
    dplyr::group_by(Season) %>%
    dplyr::summarise(tot_played = n())

# We can also do this to get a crude average games per season
dim(games_data)[1]/length(all_seasons)

get_single_season_data <- function(games_data, rseason){

    #-------------------------------------------------------------------
    # Basic checks on Data
    #-------------------------------------------------------------------

    # Get only the games for the required season
    games_data_rseason <- games_data %>%
        dplyr::filter(Season == rseason)


    #-------------------------------------------------------------------
    # CREATE CORE DATA for a SEASON
    #-------------------------------------------------------------------

    # We want to get an ordering of teams and a uniq team number for
    # the season
    unq_teams <- games_data_rseason %>%
        dplyr::select(home) %>%
        dplyr::distinct(.) %>%
        dplyr::rename(team = home) %>%
        dplyr::arrange(team) %>%
        dplyr::mutate(ind = 1:n())

    # Get Home teams as the primary "team" column
    games_home <- games_data_rseason %>%
        dplyr::rename(team = home,
                      team_other = away,
                      team_score = homescore,
                      team_other_score = awayscore) %>%
        dplyr::select(GameID, date, team, team_other,
                      team_score, team_other_score,
                      Season)

    # Get Away teams as the primary "team" column
    games_away <- games_data_rseason %>%
        dplyr::rename(team = away,
                      team_other = home,
                      team_score = awayscore,
                      team_other_score = homescore) %>%
        dplyr::select(GameID, date, team, team_other,
                      team_score, team_other_score,
                      Season)

    # Combine into a single dataframe and create the "round" for each team
    # based on play date
    games_all <- games_home %>%
        dplyr::bind_rows(games_away) %>%
        dplyr::group_by(team) %>%
        dplyr::mutate(round = row_number(date))

    # Join on the unique team id number created for the "primary" team
    games_all <- games_all %>%
        dplyr::left_join(x = ., y = unq_teams, by = c("team")) %>%
        dplyr::rename(team_ind = ind)

    # Join on the unique team id number created for the "other" team
    # diff indicator - currently gives 0 value for a tie
    # TODO: Check how we will deal with ties in our model

    games_all <- games_all %>%
        dplyr::left_join(x = ., y = unq_teams, by = c("team_other" = "team")) %>%
        dplyr::rename(team_other_ind = ind) %>%
        dplyr::mutate(diff = team_score - team_other_score,
                      diff_sign = sign(diff))

    #-------------------------------------------------------------------
    # CREATE CARTESIAN pairwise master dataframe
    #-------------------------------------------------------------------

    all_rounds <- games_all %>%
        dplyr::ungroup() %>%
        dplyr::select(round) %>%
        dplyr::distinct() %>%
        dplyr::arrange(round)

    all_team_ind <- unq_teams %>%
        dplyr::ungroup() %>%
        dplyr::select(ind) %>%
        dplyr::distinct() %>%
        dplyr::arrange(ind) %>%
        dplyr::rename(team_ind = ind)

    all_team_other_ind <- unq_teams %>%
        dplyr::ungroup() %>%
        dplyr::select(ind) %>%
        dplyr::distinct() %>%
        dplyr::arrange(ind) %>%
        dplyr::rename(team_other_ind = ind)

    # cartesian product
    all_round_team_combs <- all_rounds %>%
        tidyr::crossing(all_team_ind) %>%
        tidyr::crossing(all_team_other_ind)

    out_list <- list("games_all" = games_all,
                     "all_round_team_combs" = all_round_team_combs)

    base::return(out_list)
}

get_pairwise_diff_single_round <- function(single_season_data,
                                           round_num){

    # Get the diffs for a single round for all teams
    games_round <- single_season_data$games_all %>%
        dplyr::ungroup() %>%
        dplyr::filter(round == round_num) %>%
        dplyr::select(team_ind, team_other_ind, diff, diff_sign) %>%
        dplyr::arrange(team_ind)

    # Get a dataframe of all possible pairwise differences across teams
    games_round_all_combs <- single_season_data$all_round_team_combs %>%
        dplyr::ungroup() %>%
        dplyr::filter(round == round_num) %>%
        dplyr::select(team_ind, team_other_ind) %>%
        dplyr::left_join(x = .,
                         y = games_round,
                         by = c("team_ind" = "team_ind",
                                "team_other_ind" = "team_other_ind"))

    out_df <- games_round_all_combs %>%
        dplyr::select(-diff_sign)

    base::return(out_df)
}

write_csv_diffs <- function(rseason, round_num, all_rounds_rseason){
    out_nfl_dir <- here::here("data", "nfl", rseason)
    out_file_name <- stringr::str_c("round",
                                    stringr::str_pad(round_num,
                                                     width = 2,
                                                     side = "left",
                                                     pad = "0"),
                                    sep = "_") %>%
        stringr::str_c(., ".csv")
    out_file_path <- base::file.path(out_nfl_dir, out_file_name)
    readr::write_csv(x = all_rounds_rseason[[round_num]],
                     path = out_file_path)
}

create_single_season_csv <- function(rseason, round_nums){
    dat_rseason <- get_single_season_data(games_data = games_data,
                                          rseason = rseason)

    all_rounds_rseason <- purrr::map(round_nums,
                                     ~get_pairwise_diff_single_round(single_season_data = dat_rseason,
                                                                     round_num = .x))

    # Write out all rounds for the season
    purrr::walk(.x = round_nums,
                ~write_csv_diffs(rseason = rseason,
                                 round_num = .x,
                                 all_rounds_rseason=all_rounds_rseason))
}

#-------------------------------------------------------------------
# Run code for all seasons and rounds
#-------------------------------------------------------------------

# Define all seasons and rounds (16 per season)
rseasons <- 2009:2016
round_nums <- 1:16

# Run for all seasons and rounds
purrr::walk(.x = rseasons,
            ~create_single_season_csv(rseason = .x, round_nums = round_nums))