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
               "broom",
               "broom.mixed", # mixed effects functions, e.g. augment for mixed effects models
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
               "dplyr", # e.g. pipe functions
               "purrr", 
               "rstatix", 
               "stringr", 
               "tidyr", # e.g. wide to long transform
               "performance") 
  # format = "qs", # Optionally set the default storage format. qs is fast.
)

# Environment: save package to logfile
renv::snapshot()

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
source("R/functions.R")

# tar_source("other_functions.R") # Source other scripts as needed.

list(
  ## define some variables in targets TODO: targets doesnt accept it, maybe use as normal list
  # tar_target(
  #   experiments,
  #   command=as.list(c("ERN","LRP","MMN","N170","N2pc","N400","P3")),
  #   iteration = "list"
  # ),
  # tar_target(
  #   model_types,
  #   c("EEGNET","Sliding Window")
  # ),
  # 
  
  ## define raw data files
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
  
  ## import and recode datasets
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
  
  ## Example results of Luck forking path
  tar_target(
    name = timeresolved_luck,
    command = timeresolved_plot(data_sliding)
  ),
  
  ## Overview of decoding accuracies for each pipeline
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
  
  # TODO: HLM simulations in pipeline?
  
  ## GROUPINGS
  tar_group_by(
    data_eegnet_exp, 
    data_eegnet, 
    experiment,
  ),    
  tar_group_by(
    data_tsum_exp, 
    data_tsum, 
    experiment 
  ),  
  
  ## HLM for EEGNET
  
  tar_target(
    name=eegnet_HLM,
    command=lme4::lmer(formula="accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment + ( ref + hpf + lpf + emc + mac + base + det + ar + experiment | subject)",
                 control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
                 data = data_eegnet)
  ),
  tar_target(
    name = eegnet_HLM_exp,
    command=lme4::lmer(formula="accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)",
                       control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
                       data = data_eegnet_exp),
    pattern = map(data_eegnet_exp),
    iteration = "list"
  ),  

  ## LM for sliding
  tar_target(
    name=sliding_LM,
    command=lm(formula="tsum ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment",
               data = data_tsum)
  ),
  tar_target(
    name = sliding_LM_exp,
    command=lm(formula="tsum ~ ref + hpf + lpf + emc + mac + base + det + ar",
               data = data_tsum_exp),
    pattern = map(data_tsum_exp),
    iteration = "list"
  ),
  
  ## Estimated marginal means
  
  ### EEGNET
  tar_target(
    name=eegnet_HLM_emm,
    command=est_emm(eegnet_HLM, 
                    variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
  ),
  # split means and contrasts
  tar_target(eegnet_HLM_emm_means, eegnet_HLM_emm[[1]]),
  tar_target(eegnet_HLM_emm_contrasts, eegnet_HLM_emm[[2]]),
  tar_target(eegnet_HLM_emm_omni, eegnet_HLM_emm[[3]]),
  
  # TODO: var explained of random slopes and random subject intercepts?
  
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
  tar_target(eegnet_HLM_exp_emm_omni, eegnet_HLM_exp_emm[[3]], pattern=map(eegnet_HLM_exp_emm), iteration="list"),
  
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
  tar_target(eegnet_HLM_exp_emm_omni_ungrouped,
             command = ungrouping(eegnet_HLM_exp_emm_omni)),
  tar_target(eegnet_HLM_emm_omni_comb,
             command = {
               combine_single_whole(eegnet_HLM_exp_emm_omni_ungrouped,
                                    eegnet_HLM_emm_omni) %>%
                 mutate(p.fdr = p.adjust(.$p.value, "BY", length(.$p.value))) %>%
                 mutate(sign.unc = stars.pval(.$p.value)) %>%
                 mutate(sign.fdr = stars.pval(.$p.fdr))
               }
             ),  
  
  ### SLIDING
  tar_target(
    name=sliding_LM_emm,
    command=est_emm(sliding_LM, 
                    variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
  ),
  tar_target(sliding_LM_emm_means, sliding_LM_emm[[1]]),
  tar_target(sliding_LM_emm_contrasts, sliding_LM_emm[[2]]),
  tar_target(sliding_LM_emm_omni, sliding_LM_emm[[3]]),
  
  tar_target(
    name=sliding_LM_exp_emm,
    command=est_emm(sliding_LM_exp, 
                    variables = c("ref","hpf","lpf","emc","mac","base","det","ar")),
    pattern = map(sliding_LM_exp),
    iteration = "list"
  ),
  tar_target(sliding_LM_exp_emm_means, sliding_LM_exp_emm[[1]], pattern=map(sliding_LM_exp_emm), iteration="list"),
  tar_target(sliding_LM_exp_emm_contrasts, sliding_LM_exp_emm[[2]], pattern=map(sliding_LM_exp_emm), iteration="list"),
  tar_target(sliding_LM_exp_emm_omni, sliding_LM_exp_emm[[3]], pattern=map(sliding_LM_exp_emm), iteration="list"),
  
  # TODO Omni, also other models, and maybe combine it with the targets using dyn branching?
  
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
  tar_target(sliding_LM_exp_emm_omni_ungrouped,
             command = ungrouping(sliding_LM_exp_emm_omni)),
  tar_target(sliding_LM_emm_omni_comb,
             command = {
               combine_single_whole(sliding_LM_exp_emm_omni_ungrouped,
                                    sliding_LM_emm_omni) %>%
                 mutate(p.fdr = p.adjust(.$p.value, "BY", length(.$p.value))) %>%
                 mutate(sign.unc = stars.pval(.$p.value)) %>%
                 mutate(sign.fdr = stars.pval(.$p.fdr))
             }
  ),    
  
  
  
  
  ## heatmaps of EMMs
  # TODO: use the omni test significances to highlight the facets
  tar_target(eegnet_heatmap,
            command=heatmap(eegnet_HLM_emm_means_comb)),
  tar_target(sliding_heatmap,
             heatmap(sliding_LM_emm_means_comb)),
  tar_target(
    name = heatmaps,
    command = {
      ggarrange(eegnet_heatmap + labs(title="EEGNET"), 
                sliding_heatmap + labs(title="Sliding Window"), 
                labels = c("A", "B"),
                ncol = 1, nrow = 2)
    }),
  
  ## ECDF with best models marked
  

    
  ## concatenate all models and datas in one target
  tar_target( 
    name=models_combined,
    command=c(list(eegnet_HLM), eegnet_HLM_exp, list(sliding_LM), sliding_LM_exp)
  ), 
  # tar_target( # this does not work !!
  #   name=data_combined,
  #   command=c(list(data_eegnet), data_eegnet_exp_list, list(data_tsum), data_tsum_exp_list)
  # ), 
  
  ## diagnostics for all models (HLM, LM, ALL and experiment wise)
  ### convergence check
  tar_target( 
    name=convergence_checks,
    command=check_convergence(models_combined),
    pattern=map(models_combined),
    iteration="list"
  ), 
  
  ### qq plots
  #### EEGNET
  tar_target(eegnet_HLM_qq,
    command = qqplot(model=eegnet_HLM, data=data_eegnet)),
  tar_target(eegnet_HLM_exp_qq,
    command = qqplot(model=eegnet_HLM_exp, data=data_eegnet_exp),
    pattern = map(eegnet_HLM_exp, data_eegnet_exp),
    iteration ="list"),
  tar_target(eegnet_HLM_exp_qq_agg,
    command = eegnet_HLM_exp_qq),
  tar_target(eegnet_HLM_qq_comb,
    {plt <- ggarrange(plotlist = c(list(eegnet_HLM_qq),eegnet_HLM_exp_qq_agg))
     annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - EEGNET", 
                     color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_qq,
             command = qqplot(model=sliding_LM, data=data_tsum)),
  tar_target(sliding_LM_exp_qq,
    qqplot(model=sliding_LM_exp, data=data_tsum_exp),
    pattern = map(sliding_LM_exp, data_tsum_exp),
    iteration ="list"),
  tar_target(sliding_LM_exp_qq_agg,
    command = sliding_LM_exp_qq),
  tar_target(sliding_LM_qq_comb,
    {plt <- ggarrange(plotlist = c(list(sliding_LM_qq),sliding_LM_exp_qq_agg))
    annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - Sliding", 
                    color = "black", face = "bold", size = 16))}),
  
  ### res_vs_fitted plots
  #### EEGNET
  tar_target(eegnet_HLM_rvf,
             command = rvfplot(model=eegnet_HLM, data=data_eegnet)),
  tar_target(eegnet_HLM_exp_rvf,
             command = rvfplot(model=eegnet_HLM_exp, data=data_eegnet_exp),
             pattern = map(eegnet_HLM_exp, data_eegnet_exp),
             iteration ="list"),
  tar_target(eegnet_HLM_exp_rvf_agg,
             command = eegnet_HLM_exp_rvf),
  tar_target(eegnet_HLM_rvf_comb,
             {plt <- ggarrange(plotlist = c(list(eegnet_HLM_rvf),eegnet_HLM_exp_rvf_agg))
             annotate_figure(plt, top = text_grob("Residual vs. Fitted Plots - EEGNET", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_rvf,
             command = rvfplot(model=sliding_LM, data=data_tsum)),
  tar_target(sliding_LM_exp_rvf,
             rvfplot(model=sliding_LM_exp, data=data_tsum_exp),
             pattern = map(sliding_LM_exp, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LM_exp_rvf_agg,
             command = sliding_LM_exp_rvf),
  tar_target(sliding_LM_rvf_comb,
             {plt <- ggarrange(plotlist = c(list(sliding_LM_rvf),sliding_LM_exp_rvf_agg))
             annotate_figure(plt, top = text_grob("Residual vs. Fitted Plots - Sliding", 
                             color = "black", face = "bold", size = 16))}),

  ### sqrt abs std res_vs_fitted plots
  #### EEGNET
  tar_target(eegnet_HLM_sasrvf,
             command = sasrvfplot(model=eegnet_HLM, data=data_eegnet)),
  tar_target(eegnet_HLM_exp_sasrvf,
             command = sasrvfplot(model=eegnet_HLM_exp, data=data_eegnet_exp),
             pattern = map(eegnet_HLM_exp, data_eegnet_exp),
             iteration ="list"),
  tar_target(eegnet_HLM_exp_sasrvf_agg,
             command = eegnet_HLM_exp_sasrvf),
  tar_target(eegnet_HLM_sasrvf_comb,
             {plt <- ggarrange(plotlist = c(list(eegnet_HLM_sasrvf),eegnet_HLM_exp_sasrvf_agg))
             annotate_figure(plt, top = text_grob("Scale-Location-Plots - EEGNET", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_sasrvf,
             command = sasrvfplot(model=sliding_LM, data=data_tsum)),
  tar_target(sliding_LM_exp_sasrvf,
             sasrvfplot(model=sliding_LM_exp, data=data_tsum_exp),
             pattern = map(sliding_LM_exp, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LM_exp_sasrvf_agg,
             command = sliding_LM_exp_sasrvf),
  tar_target(sliding_LM_sasrvf_comb,
             {plt <- ggarrange(plotlist = c(list(sliding_LM_sasrvf),sliding_LM_exp_sasrvf_agg))
             annotate_figure(plt, top = text_grob("Scale-Location-Plots - Sliding", 
                             color = "black", face = "bold", size = 16))}),
  
  
  ## RFX Boxplots
  tar_target(eegnet_RFX_all,
             rfx_vis(eegnet_HLM, data_eegnet)),
  tar_target(eegnet_RFX_exp,
             rfx_vis(eegnet_HLM_exp, data_eegnet_exp),
             pattern=map(eegnet_HLM_exp, data_eegnet_exp),
             iteration="list"),
  tar_target(eegnet_RFX_exp_agg,
             eegnet_RFX_exp),
  tar_target(eegnet_RFX_plot,
             {
               plt <- ggarrange(plotlist = c(list(eegnet_RFX_all),eegnet_RFX_exp_agg))
               # plt <- ggarrange(eegnet_RFX_all + theme(plot.margin = margin(0, 0, 0, 0)), # Remove margins from plot1
               #                  ggarrange(plotlist=eegnet_RFX_exp_agg, 
               #                            ncol = 2, nrow = 5), 
               #        # Arrange remaining plots in one column
               #                  ncol = 1, nrow=4)
               annotate_figure(plt, top = text_grob("Random Effects - EEGNET", 
                               color = "black", face = "bold", size = 16))
             })
  
#  tar_target(
#    name=eegnet_HLM_simulations,
#    command=lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + experiment + (1 | subject)",
#                 control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
#                 data = data_eegnet)
#  ),
  
  # TODO: i could synchronize the analyses of eegnet/sliding by just using pattern instead of two targets? (no group by)

# TODO: add another dataset (infants)

## OLD ################################################################

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

# tar_target(
#   name = marginal_means,
#   command = estimate_marginal_means(data_eegnet, 
#                                     variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment"))
#   #pattern = map(data_dataset)
# ),
# tar_target(
#   name=stats_all,
#   command=paired_tests(marginal_means, study=unique(data_dataset$dataset)[1])
#   #pattern = map(marginal_means, data_dataset),
#   #iteration = "list"
# ),
# tar_target(
#   name=raincloud_all,
#   command = raincloud_mm(marginal_means, title="ERPCORE") #unique(data_dataset$dataset)),
#   #pattern = map(marginal_means, data_dataset),
#   #iteration = "list"
# ),
# tar_target(
#   name=paired_all,
#   command = paired_box(marginal_means, title="ERPCORE") #unique(data_dataset$dataset)) #,
#   #pattern = map(marginal_means, data_dataset),
#   #iteration = "list"
# ),

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


)



