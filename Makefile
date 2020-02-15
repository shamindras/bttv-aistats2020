.PHONY : conda_bttv conda_rem_reinst_bttv get_nascar_data get_nfl_filt_data clean

# Create the various conda environments
setup_r_packages:
	Rscript R/setup_r_packages.R

conda_bttvaistats2020:
	conda env create -f=./conda_envs/bttvaistats2020/environment.yml

conda_rem_reinst_bttvaistats2020:
	conda remove --name bttvaistats2020 --all
	conda_bttvaistats2020

get_nfl_filt_data:
	Rscript R/get_nfl_data_filt_teams.R

clean:
	find R  -iname '*.DS_Store' -print0 | xargs -0 rm -rf
	find R  -iname '.Rhistory' -print0 | xargs -0 rm -rf
	find R  -iname '*.nb.html' -print0 | xargs -0 rm -rf
	find R  -iname '*_files' -print0 | xargs -0 rm -rf
	find R  -iname '*_cache' -print0 | xargs -0 rm -rf
