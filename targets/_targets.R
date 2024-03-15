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
  packages = c("dplyr", "ggplot2", "ggsignif", "readr", "lmerTest", "emmeans", "magrittr", "ggpubr", "data.table",
               "tidyverse", "tidyquant", "ggdist", "ggthemes", "broom", "dplyr", "purrr", "rstatix", "tidyr") # packages that your targets use
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
  # define raw data files
  tar_target(
    name = eegnet_file,
    command = "eegnet.csv",
    format = "file"
  ),
  tar_target(
    name = sliding_file,
    command = "sliding.csv",
    format = "file"
  ),
  tar_target(
    name = tsum_file,
    command = "sliding_tsums.csv",
    format = "file"
  ),
  
  # import and recode datasets
  tar_target(
    name = data_eegnet,
    command = get_preprocess_data(eegnet_file)
  ),
  # for now, SLIDING only in ERPCORE, because MIPDB has errors, TODO later: tar_group_by and include MIPDB
  tar_target(
    name = data_sliding,
    command = {get_preprocess_data(sliding_file) %>% filter(dataset == "erpcore") %>% select(-c(forking_path, dataset))}
  ),
  tar_target(
    name = data_tsum,
    command = {get_preprocess_data(tsum_file) %>% filter(dataset == "erpcore") %>% select(-c(forking_path, dataset))}
  ),
  
  #### Example results of Luck forking path
  tar_target(
    name = timeresolved_luck,
    command = timeresolved_plot(data_sliding)
  ),
  
  #### sliding processing ####
  
  # calculation of marginal means (total and per experiment)
  tar_target(
    name = results_sliding,
    command = estimate_marginal_means_sliding(data_tsum, per_exp=FALSE)
  ),
  tar_target(
    name = results_sliding_experiment,
    command = estimate_marginal_means_sliding(data_tsum, per_exp=TRUE)
  ),
  
  # calculation of t values
  
  
  # plotting
  tar_target(
    name = plot_sliding,
    command = sliding_plot_all(results_sliding)
  ),
  tar_target(
    name = plot_sliding_experiment,
    command = sliding_plot_experiment(results_sliding_experiment)
  ),
  tar_target(
    name = plot_ecdf,
    command = ecdf(data_tsum)
  ),
  
  
  
  
  #### eegnet processing ####
  tar_group_by(
    data_dataset, 
    data_eegnet, 
    dataset # this groups the dataframe by experiment, for later single evaluation
  ),  

  tar_target(
    name = marginal_means,
    command = estimate_marginal_means(data_dataset, 
                                      variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment")),
    pattern = map(data_dataset)
  ),
  tar_target(
    name=stats_all,
    command=paired_tests(marginal_means, study=unique(data_dataset$dataset)[1]),
    pattern = map(marginal_means, data_dataset),
    iteration = "list"
  ),
  tar_target(
    name=raincloud_all,
    command = raincloud_mm(marginal_means, title=unique(data_dataset$dataset)),
    pattern = map(marginal_means, data_dataset),
    iteration = "list"
  ),
  tar_target(
    name=paired_all,
    command = paired_box(marginal_means, title=unique(data_dataset$dataset)),
    pattern = map(marginal_means, data_dataset),
    iteration = "list"
  ),

  # single analyses per experiment
  tar_group_by(
    single_exp_data,
    data_dataset,
    experiment # this groups the dataframe by experiment, for later single evaluation
  ),
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



