# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes) # e.g. for tar_map
# Load other packages as needed.

# Set target options:
tar_option_set(
  packages = c("dplyr", "ggplot2", "ggsignif", "readr", "lmerTest", "emmeans", "magrittr", "ggpubr",
               "tidyverse", "tidyquant", "ggdist", "ggthemes", "broom", "dplyr", "purrr", "rstatix") # packages that your targets use
  # format = "qs", # Optionally set the default storage format. qs is fast.
  #
  # Pipelines that take a long time to run may benefit from
  # optional distributed computing. To use this capability
  # in tar_make(), supply a {crew} controller
  # as discussed at https://books.ropensci.org/targets/crew.html.
  # Choose a controller that suits your needs. For example, the following
  # sets a controller that scales up to a maximum of two workers
  # which run as local R processes. Each worker launches when there is work
  # to do and exits if 60 seconds pass with no tasks to run.
  #
  #   controller = crew::crew_controller_local(workers = 2, seconds_idle = 60)
  #
  # Alternatively, if you want workers to run on a high-performance computing
  # cluster, select a controller from the {crew.cluster} package.
  # For the cloud, see plugin packages like {crew.aws.batch}.
  # The following example is a controller for Sun Grid Engine (SGE).
  # 
  #   controller = crew.cluster::crew_controller_sge(
  #     # Number of workers that the pipeline can scale up to:
  #     workers = 10,
  #     # It is recommended to set an idle time so workers can shut themselves
  #     # down if they are not running tasks.
  #     seconds_idle = 120,
  #     # Many clusters install R as an environment module, and you can load it
  #     # with the script_lines argument. To select a specific verison of R,
  #     # you may need to include a version string, e.g. "module load R/4.3.2".
  #     # Check with your system administrator if you are unsure.
  #     script_lines = "module load R"
  #   )
  #
  # Set other options as needed.
)

# Environment: save package to logfile
renv::snapshot()

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
source("R/functions.R")

# tar_source("other_functions.R") # Source other scripts as needed.

#experiments = c("ERN","LRP","MMN","N170","N2pc","N400","P3")

# Replace the target list below with your own:
list(
  tar_target(
    name = file,
    #command = "ERPCORE_eegnet.csv",
    command = "eegnet.csv",
    format = "file"
  ),
  tar_target(
    name = data,
    command = get_preprocess_data(file)
  ),
  tar_target(
    name = marginal_means,
    command = estimate_marginal_means(data, 
                                      variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
  ),
  tar_target(
    name=stats_all,
    command=paired_tests(marginal_means)
  ),
  tar_target(
    name=raincloud_all,
    command = raincloud_mm(marginal_means)
  ),
  tar_target(
    name=paired_all,
    command = paired_box(marginal_means)
  ),
  
  # single analyses per experiment
  tar_group_by(
    single_exp_data, 
    data, 
    experiment # this groups the dataframe by experiment, for later single evaluation
  ),
  
    # gets the experiment names, as i can not find it in some metadata
  #  tar_target(
  #    exp_names,
  #    command = {
  #      unique(single_exp_data$experiment),
  #      
  #      
  #      
  #    },
  #    pattern = map(single_exp_data)
  
  tar_target(
    name = marginal_means_single_exp,
    command = estimate_marginal_means(single_exp_data, 
                                      variables = c("ref","hpf","lpf","base","det","ar","emc","mac")),
    pattern = map(single_exp_data)
  ),
  tar_target(
    name=stats_single,
    command=paired_tests(marginal_means_single_exp),
    pattern = map(marginal_means_single_exp),
    iteration = "list"
  ),
  tar_target(
    name=raincloud_single,
    command = raincloud_mm(marginal_means_single_exp, 
                           title=unique(single_exp_data$experiment)),
    pattern = map(marginal_means_single_exp, single_exp_data),
    iteration = "list"
  ),
  tar_target(
    name=paired_single,
    command = paired_box(marginal_means_single_exp,
                         title = unique(single_exp_data$experiment)),
    pattern = map(marginal_means_single_exp, single_exp_data),
    iteration = "list"
  )
)

# # TODO: check model assumptions HLM: https://stats.stackexchange.com/questions/376273/assumptions-for-lmer-models
# 
# # tar_target(
# #   name = model,
# #   command = fit_model(data)
# #   #description = "Regression of ozone vs temp" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
# # ),
# #description = "Regression of ozone vs temp" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
# 
# # tar_target(
# #   name = model,
# #   command = fit_model(data)
# #   #description = "Regression of ozone vs temp" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
# # ),
# # tar_target(
# #   name = plot,
# #   command = plot_model(model, data)
# #   #description = "Scatterplot of model & data" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
# # )
# 
# 
# # tar_target(
# #   name = hlm_all,
# #   command = lmer(data=data, formula="accuracy ~ ref + hpf + lpf + base + det + ar + emc + mac + experiment + (1 | subject)")
# #   #(formula, data = data)
# # ),
# # tar_target(
# #   name = emm_all_exp,
# #   command = est_emm(hlm_all, variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
# # ),
# #tar_target(
# #  name = emm_all_exp_plots,
# #  command = plot_emm(emm_all_exp, feature="experiment") # or without ""
# #),
# 
# 
# 
# # tar_target(hlm_single, 
# #            hlm(single_exp_data, formula = "accuracy ~ ref + hpf + lpf + base + det + ar + emc + mac + (1 | subject)"), 
# #            pattern = map(single_exp_data),
# #            iteration = "list" # make each a list element, else it can not be read properly later, but maybe without, it can better be processed/merged later? TODO Test
# #            )
# 
# # tar_target(single_coefs, 
# #            get_ffx_coefs(hlm_single), 
# #            pattern = map(hlm_single), #single_exp_data
# #            iteration = "list" # make each a list element, else it can not be read properly later, but maybe without, it can better be processed/merged later? TODO Test
# #)
# 
# 
# #tar_target(
# #  name = filtered_data,
# #  command = filter_experiment(factorized_data)
# #)
