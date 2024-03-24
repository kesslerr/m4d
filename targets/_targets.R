# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes) # e.g. for tar_map
library(future) # incl for parallel processing
plan(multisession)

# Load other packages as needed.

# Set target options:
tar_option_set( # packages that your targets use
  packages = c("colorspace", # nice colormaps
               "dplyr", 
               "ggplot2", 
               "ggsignif", 
               "ggpattern", # allows to make patterns into tiles or bars
               "gtools", # to convert p values into asterisks
               "ggpubr", # ggplots publication ready, e.g. labeling plots and arranging mutliple plots
               "readr", 
               #"lmerTest", # has p value estimations
               "lme4", # seems to be 20% faster
               "emmeans", 
               "magrittr", 
               "ggpubr", 
               "data.table",
               "tidyverse", 
               "tidyquant", 
               "ggdist", 
               "ggthemes", 
               "broom", 
               "dplyr", 
               "purrr", 
               "rstatix", 
               "stringr", 
               "tidyr",
               "performance") 
  # format = "qs", # Optionally set the default storage format. qs is fast.
)

# Environment: save package to logfile
renv::snapshot()

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
source("R/functions.R")

# tar_source("other_functions.R") # Source other scripts as needed.

experiments = c("ERN","LRP","MMN","N170","N2pc","N400","P3")

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

  # for now, only in ERPCORE, because MIPDB has errors, TODO later: tar_group_by and include MIPDB
  tar_target(
    name = data_eegnet,
    command = {get_preprocess_data(eegnet_file) %>% filter(dataset == "ERPCORE") %>% select(-c(dataset))} #forking_path, 
  ),
  tar_target(
    name = data_sliding,
    command = {get_preprocess_data(sliding_file) %>% filter(dataset == "ERPCORE") %>% select(-c(forking_path, dataset))} # 
  ),
  tar_target(
    name = data_tsum,
    command = {get_preprocess_data(tsum_file) %>% filter(dataset == "ERPCORE") %>% select(-c(forking_path, dataset))} #forking_path, 
  ),
  
  #### Example results of Luck forking path
  tar_target(
    name = timeresolved_luck,
    command = timeresolved_plot(data_sliding)
  ),
  
  #### Overview of decoding accuracies for each pipeline
  tar_target(
    name = overview_accuracy,
    command = raincloud_acc(data_eegnet, title = "EEGNET")
  ),
  tar_target(
    name = overview_tsum,
    command = raincloud_acc(data_tsum, title = "Sliding Window")
  ),
  tar_target(
    name = overview,
    command = {
      ggarrange(overview_accuracy, overview_tsum, 
                labels = c("A", "B"),
                ncol = 1, nrow = 2)
    }
  ),  
  #### sliding processing ####
  
  # calculation of marginal means (total and per experiment)
  #tar_target(
  #  name = results_sliding,
  #  command = estimate_marginal_means_sliding(data_tsum, per_exp=FALSE)
  #),
  #tar_target(
  #  name = results_sliding_experiment,
  #  command = estimate_marginal_means_sliding(data_tsum, per_exp=TRUE)
  #),
  
  # calculation of t values
  
  
  # plotting
  #tar_target(
  #  name = plot_sliding,
  #  command = sliding_plot_all(results_sliding)
  #),
  #tar_target(
  #  name = plot_sliding_experiment,
  #  command = sliding_plot_experiment(results_sliding_experiment)
  #),
  #tar_target(
  #  name = plot_ecdf,
  #  command = ecdf(data_tsum)
  #),
  
  
  
  
  #### eegnet processing ####
  #tar_group_by(
  #  data_dataset, 
  #  data_eegnet, 
  #  dataset # this groups the dataframe by experiment, for later single evaluation
  #),  

  tar_target(
    name = marginal_means,
    command = estimate_marginal_means(data_eegnet, 
                                      variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
    #pattern = map(data_dataset)
  ),
  tar_target(
    name=stats_all,
    command=paired_tests(marginal_means, study=unique(data_dataset$dataset)[1])
    #pattern = map(marginal_means, data_dataset),
    #iteration = "list"
  ),
  tar_target(
    name=raincloud_all,
    command = raincloud_mm(marginal_means, title="ERPCORE") #unique(data_dataset$dataset)),
    #pattern = map(marginal_means, data_dataset),
    #iteration = "list"
  ),
  tar_target(
    name=paired_all,
    command = paired_box(marginal_means, title="ERPCORE") #unique(data_dataset$dataset)) #,
    #pattern = map(marginal_means, data_dataset),
    #iteration = "list"
  ),

  # single analyses per experiment
  # tar_group_by(
  #   single_exp_data,
  #   data_dataset,
  #   experiment # this groups the dataframe by experiment, for later single evaluation
  # ),
  # tar_target(
  #   name = marginal_means_single_exp,
  #   command = estimate_marginal_means(single_exp_data,
  #                                     variables = c("ref","hpf","lpf","base","det","ar","emc","mac")),
  #   pattern = map(single_exp_data)
  # ),
  # tar_target(
  #   name=stats_single,
  #   command=paired_tests(marginal_means_single_exp),
  #   pattern = map(marginal_means_single_exp),
  #   iteration = "list"
  # ),
  # tar_target(
  #   name=raincloud_single,
  #   command = raincloud_mm(marginal_means_single_exp,
  #                          title=unique(single_exp_data$experiment)),
  #   pattern = map(marginal_means_single_exp, single_exp_data),
  #   iteration = "list"
  # ),
  # tar_target(
  #   name=paired_single,
  #   command = paired_box(marginal_means_single_exp,
  #                        title = unique(single_exp_data$experiment)),
  #   pattern = map(marginal_means_single_exp, single_exp_data),
  #   iteration = "list"
  # ),
  
  # HLM + HLM simulations  
  
  ## HLM
  
  tar_target(
    name=eegnet_HLM,
    command=lme4::lmer(formula="accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment + ( ref + hpf + lpf + emc + mac + base + det + ar + experiment | subject)",
                 control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
                 data = data_eegnet)
  ),
  tar_target(
    name=eegnet_HLM_check,
    command=check_convergence(eegnet_HLM)
  ),
  tar_target(
    name=eegnet_HLM_qq,
    command = qqplot.data(model=eegnet_HLM, 
                          data="",
                          title="ALL")
  ),
  # marginal means
  tar_target(
    name=eegnet_HLM_emm,
    command=est_emm(eegnet_HLM, 
                    variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
  ),
  # split means and contrasts
  tar_target(eegnet_HLM_emm_means, eegnet_HLM_emm[[1]]),
  tar_target(eegnet_HLM_emm_contrasts, eegnet_HLM_emm[[2]]),
  
  # TODO: var explained of random slopes and random subject intercepts?
  
  ## for each experiment
  tar_group_by(
    data_eegnet_exp, 
    data_eegnet, 
    experiment # this groups the dataframe by experiment, for later single evaluation
  ),  
  tar_target(
    name = eegnet_HLM_exp,
    command=lme4::lmer(formula="accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)",
                 control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
                 data = data_eegnet_exp),
    pattern = map(data_eegnet_exp),
    iteration = "list"
  ),

  tar_target(
    name=eegnet_HLM_exp_check,
    command=check_convergence(eegnet_HLM_exp),
    pattern = map(eegnet_HLM_exp),
    iteration = "list"
  ),
  tar_target(
    name=eegnet_HLM_exp_qq,
    command = qqplot.data(model=eegnet_HLM_exp, 
                          data=data_eegnet_exp,
                          title=""),
    pattern = map(eegnet_HLM_exp, data_eegnet_exp),
    iteration ="list" # "vector"
  ),
  tar_target(
    name=eegnet_HLM_exp_qq_agg,
    command = eegnet_HLM_exp_qq
  ),
  tar_target(
    name=eegnet_HLM_qq_comb,
    command = {ggarrange(plotlist = c(list(eegnet_HLM_qq),eegnet_HLM_exp_qq_agg))},
  ),
  
  tar_target(
    name=eegnet_HLM_exp_emm,
    command=est_emm(eegnet_HLM_exp, 
                    variables = c("ref", "hpf","lpf","emc","mac","base","det","ar")),
    pattern = map(eegnet_HLM_exp),
    iteration = "list"
  ),
  # split means and contrasts
  tar_target(eegnet_HLM_exp_emm_means, eegnet_HLM_exp_emm[[1]], pattern=map(eegnet_HLM_exp_emm), iteration="list"),
  tar_target(eegnet_HLM_exp_emm_contrasts, eegnet_HLM_exp_emm[[2]], pattern=map(eegnet_HLM_exp_emm), iteration="list"),
  
  # combine ALL and EXP
  tar_target(eegnet_HLM_exp_emm_means_ungrouped,
             command = ungrouping(eegnet_HLM_exp_emm_means)),
  tar_target(eegnet_HLM_emm_means_comb,
             command = combine_single_whole(eegnet_HLM_exp_emm_means_ungrouped,
                                            eegnet_HLM_emm_means)),
  tar_target(eegnet_HLM_exp_emm_contrasts_ungrouped,
             command = ungrouping(eegnet_HLM_exp_emm_contrasts)),
  tar_target(eegnet_HLM_emm_contrasts_comb,
             command = combine_single_whole(eegnet_HLM_exp_emm_contrasts_ungrouped,
                                            eegnet_HLM_emm_contrasts)),
  
  # heatmap of results
  tar_target(eegnet_heatmap,
            heatmap(eegnet_HLM_emm_means_comb)),
    
  
  #tar_combine(
  #  name = eegnet_HLM_emm_mean_comb,
  #  list(eegnet_HLM_emm_means,
  #       eegnet_HLM_exp_emm_means),
  #  pattern = map(eegnet_HLM_emm_means, eegnet_HLM_exp_emm_means),
  #  command = bind_rows(!!!.x)
  #),
  
  # TODO: merge exp and full emm results
  # TODO: heatmap of all experiments (and together) in one map, showing the MM improvement of each level
  
  
  ## LM for Sliding --> TODO, this can't be right, all signifi
  tar_target(
    name=sliding_LM,
    # TODO: are specifications correct?
    command=lm(formula="tsum ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment",
                 #control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
                 data = data_tsum)
  ),
  # marginal means
  tar_target(
    name=sliding_LM_emm,
    command=est_emm(sliding_LM, 
                    variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
  ),
  # split means and contrasts
  tar_target(sliding_LM_emm_means, sliding_LM_emm[[1]]),
  tar_target(sliding_LM_emm_contrasts, sliding_LM_emm[[2]]),
  
  
  # for each experiment
  tar_group_by(
    data_tsum_exp, 
    data_tsum, 
    experiment # this groups the dataframe by experiment, for later single evaluation
  ),  
  
  tar_target(
    name = sliding_LM_exp,
    command=lm(formula="tsum ~ ref + hpf + lpf + emc + mac + base + det + ar",
               #control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
               data = data_tsum_exp),
    pattern = map(data_tsum_exp),
    iteration = "list"
  ),
  # MM
  tar_target(
    name=sliding_LM_exp_emm,
    command=est_emm(sliding_LM_exp, 
                    variables = c("ref","hpf","lpf","emc","mac","base","det","ar")),
    pattern = map(sliding_LM_exp),
    iteration = "list"
  ),
  # split means and contrasts
  tar_target(sliding_LM_exp_emm_means, sliding_LM_exp_emm[[1]], pattern=map(sliding_LM_exp_emm), iteration="list"),
  tar_target(sliding_LM_exp_emm_contrasts, sliding_LM_exp_emm[[2]], pattern=map(sliding_LM_exp_emm), iteration="list"),
  
  # combine ALL and EXP
  tar_target(sliding_LM_exp_emm_means_ungrouped,
             command = ungrouping(sliding_LM_exp_emm_means)),
  tar_target(sliding_LM_emm_means_comb,
             command = combine_single_whole(sliding_LM_exp_emm_means_ungrouped,
                                            sliding_LM_emm_means)),
  tar_target(sliding_LM_exp_emm_contrasts_ungrouped,
             command = ungrouping(sliding_LM_exp_emm_contrasts)),
  tar_target(sliding_LM_emm_contrasts_comb,
             command = combine_single_whole(sliding_LM_exp_emm_contrasts_ungrouped,
                                            sliding_LM_emm_contrasts)),
  
  # heatmap of results
  tar_target(sliding_heatmap,
             heatmap(sliding_LM_emm_means_comb))
  
  
  
#  tar_target(
#    name=eegnet_HLM_simulations,
#    # TODO: add random slopes of all variables: hpf + lpf + emc + mac + base + det + ar + experiment
#    command=lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + experiment + (1 | subject)",
#                 control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
#                 data = data_eegnet)
#  ),
  
  # TODO: i could synchronize the analyses of eegnet/sliding by just using pattern instead of two targets? (no group by)
  
  

)



