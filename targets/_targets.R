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
               "ggpubr", # ggplots publication ready, e.g. labeling plots and arranging mutliple plots (ggarrange)
               "GGally", # e.g. pairwise correlation plots (ggpairs)
               "ggdist", 
               "ggthemes", 
               "ggsankey", # sankey plots
               "ggview", # scale plots correctly for publication
               "cowplot", # to overlay plots
               "pals", # some colorscales
               "readr", 
               #"lmerTest", # has p value estimations
               "lme4", # seems to be 20% faster
               "emmeans", 
               "JuliaCall", # use Julia from within R
               "magrittr", # for double pipe operator %<>%
               "ggpubr", 
               "data.table",
               "tidyverse", 
               "tidyquant", 
               "dplyr", # e.g. pipe functions
               "purrr", 
               "rstatix", 
               "stringr", 
               "tidyr", # e.g. wide to long transform
               "performance",
               "circlize", # chord diagram
               "xtable", # export R tables to latex files
               "stringr" # string manipulation
               ) 
  # format = "qs", # Optionally set the default storage format. qs is fast.
)

# set Julia binary
JuliaCall::julia_setup(rebuild = TRUE, # if rcall is disrupted by updates in R
                       JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")
options(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")
julia_executable <- "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/julia" # needed to call external scripts


# Environment: save package to logfile
renv::snapshot()

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
source("R/functions.R")
source("R/ampel.R")
source("R/timeresolved_baseline_artifact.R")
source("R/latex.R")
source("R/interactions.R")
source("R/rfx.R")
source("R/diagnostics.R")
source("R/ALT_pipeline.R")

table_output_dir = "../manuscript/tables/"
figure_output_dir = "../manuscript/plots/"

# EMM
#emm_options(lmerTest.limit = Inf) # TODO, include this but real command

# tar_source("other_functions.R") # Source other scripts as needed.

experiments = c("ERN","LRP","MMN","N170","N2pc","N400","P3")

list(
  ## define some variables in targets TODO: targets doesnt accept it, maybe use as normal list
  # tar_target(
  #   experiments,
  #   command=as.list(c("ERN","LRP","MMN","N170","N2pc","N400","P3")),
  #   iteration = "list"
  # ),
  # tar_target(
  #   model_types,
  #   c("EEGNET","Time-resolved")
  # ),
  # 
  
  ## define raw data files
  tar_target(
    name = eegnet_file,
    command = "../models/eegnet_extended.csv", # TODO: put this into a different folder
    format = "file"
  ),
  tar_target(
    name = sliding_file,
    command = "../models/sliding_extended.csv",
    format = "file"
  ),
  tar_target(
    name = tsum_file,
    command = "../models/sliding_tsums_extended.csv",
    format = "file"
  ),
  tar_target(
    name = sliding_avgacc_single_file,
    command = "../models/sliding_avgacc_single_extended.csv",
    format = "file"
  ),
  tar_target(
    name = demographics_file,
    command = '../data/erpcore/participants.tsv',
    format = "file"
  ),
  
  ## NEW: single LMM files
  ## first, test if multiple files can be tracked at the same time
  ## 
  # TODO: track the files maybe like this: https://stackoverflow.com/questions/69652540/how-should-i-use-targets-when-i-have-multiple-data-files
  # TODO: i could synchronize the analyses of eegnet/sliding by just using pattern instead of two targets? (no group by)
  
  ## EEGNET
  tar_target(
    name = jLMM_file_ERN,
    command = '../julia/model_ERN_eegnet.rds',
    format = "file_fast" # file_fast checks if the file is up to date!!
  ),
  tar_target(
    name = jLMM_file_LRP,
    command = '../julia/model_LRP_eegnet.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_MMN,
    command = '../julia/model_MMN_eegnet.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_N170,
    command = '../julia/model_N170_eegnet.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_N2pc,
    command = '../julia/model_N2pc_eegnet.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_N400,
    command = '../julia/model_N400_eegnet.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_P3,
    command = '../julia/model_P3_eegnet.rds',
    format = "file_fast"
  ),
  ## SLIDING
  tar_target(
    name = jLMM_file_ERN_tr,
    command = '../julia/model_ERN_time-resolved.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_LRP_tr,
    command = '../julia/model_LRP_time-resolved.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_MMN_tr,
    command = '../julia/model_MMN_time-resolved.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_N170_tr,
    command = '../julia/model_N170_time-resolved.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_N2pc_tr,
    command = '../julia/model_N2pc_time-resolved.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_N400_tr,
    command = '../julia/model_N400_time-resolved.rds',
    format = "file_fast"
  ),
  tar_target(
    name = jLMM_file_P3_tr,
    command = '../julia/model_P3_time-resolved.rds',
    format = "file_fast"
  ),
  
  tar_target(
    name = jLMM_files,
    command = c(jLMM_file_ERN, jLMM_file_LRP, jLMM_file_MMN, jLMM_file_N170, jLMM_file_N2pc, jLMM_file_N400, jLMM_file_P3),
  ),
  tar_target(name = eegnet_HLMi2, 
             readRDS(jLMM_files),
             pattern=jLMM_files,
             iteration="list"),
  tar_target(
    name = jLMM_files_tr,
    command = c(jLMM_file_ERN_tr, jLMM_file_LRP_tr, jLMM_file_MMN_tr, jLMM_file_N170_tr, jLMM_file_N2pc_tr, jLMM_file_N400_tr, jLMM_file_P3_tr),
  ),
  tar_target(name = sliding_HLMi2, 
             readRDS(jLMM_files_tr),
             pattern=jLMM_files_tr,
             iteration="list"),

  ## import and recode datasets
  tar_target(
    name = data_eegnet,
    command = {get_preprocess_data(eegnet_file)} #  %>% filter(dataset == "ERPCORE") %>% select(-c(dataset))
  ),
  tar_target(
    name = data_sliding,
    command = {get_preprocess_data(sliding_file) %>% select(-c(forking_path))} # , dataset  %>% filter(dataset == "ERPCORE") 
  ),
  tar_target(
    name = data_tsum,
    command = {get_preprocess_data(tsum_file) %>% select(-c(forking_path))} # , dataset  %>% filter(dataset == "ERPCORE") 
  ),
  tar_target(
    name = demographics,
    command = {read_tsv(demographics_file) %>% 
        # convert M to male and F to temal in variable sex
        mutate(sex = recode(sex, "M" = "Male", "F" = "Female")) %>%
        mutate_if(is.character, as.factor)
        }
  ),
  # single participant average accuracy
  tar_target(
    name = data_avgaccs,
    command = {get_preprocess_data(sliding_avgacc_single_file)} # , dataset  %>% filter(dataset == "ERPCORE")  %>% select(-c(forking_path))
  ),
  
  ## Multiverse sankey visualization
  tar_target(
    name = sankey,
    command = plot_multiverse_sankey(data_eegnet)
  ),
  tar_target(
    name = sankey_file,
    command = {
      ggsave(plot=sankey,
             filename="sankey.png",
             path=figure_output_dir,
             scale=1.5,
             width=15,
             height=5,
             units="cm",
             dpi=300)
    },
  ),
  

  ## Overview of decoding accuracies for each pipeline
  tar_target(
    name = overview_accuracy,
    command = raincloud_acc(data_eegnet, title = "EEGNet")
  ),
  tar_target( # average across subjects for each pipeline
    name = overview_accuracy_avgsub,
    command = raincloud_acc(data_eegnet %>%
                              group_by(emc, mac, lpf, hpf, ref, det, base, ar, experiment) %>% #ref, hpf, lpf, emc, mac, det, base, ar, experiment
                              summarize(accuracy = mean(accuracy)) %>% 
                              select(accuracy, everything()), # put the accuracy in the first column
                            title = "EEGNet")
  ),
  tar_target(
    name = overview_tsum,
    command = raincloud_acc(data_tsum, title = "Time-resolved")
  ),
  
  tar_target( # average across subjects for each pipeline
    name = overview_avgaccs_avgsub,
    command = raincloud_acc(data_avgaccs %>%
                              group_by(emc, mac, lpf, hpf, ref, base, det, ar, experiment) %>% #ref, hpf, lpf, emc, mac, base, det, ar, experiment
                              summarize(accuracy = mean(accuracy)) %>% 
                              select(accuracy, everything()), # put the accuracy in the first column
                            title = "Time-resolved")
  ),
  
  tar_target(
    name = overview,
    command = {
      ggarrange(overview_accuracy_avgsub, overview_tsum, 
                labels = c("A", "B"),
                ncol = 2, nrow = 1)
    }
  ), 

  tar_target(
    name = overview_poster,
    command = {
      ggarrange(overview_accuracy_avgsub, overview_avgaccs_avgsub, 
                labels = c("A", "B"),
                ncol = 1, nrow = 2)
    }
  ), 
  
  # Ampel heatmaps, to visualize ranked forking paths
  tar_target(
    name = ampel_merge,
    command = {rankampel_merge(data_eegnet, data_tsum, "EEGNet", "Time-resolved")}
  ), 
  tar_target(
    name = ampel_file,
    command = {
      ggsave(plot=ampel_merge,
             filename="ampel.png",
             path=figure_output_dir,
             scale=1,
             width=20,
             height=20,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  
  # Ampel separately
  tar_target(
    name = ampel_eegnet,
    command = {rankampel_single(data_eegnet, "EEGNet")}
  ), 
  tar_target(
    name = ampel_eegnet_file,
    command = {
      ggsave(plot=ampel_eegnet,
             filename="ampel_eegnet.png",
             path=figure_output_dir,
             scale=1.1,
             width=15,
             height=30,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  tar_target(
    name = ampel_tr,
    command = {rankampel_single(data_tsum, "Time-resolved")}
  ), 
  tar_target(
    name = ampel_tr_file,
    command = {
      ggsave(plot=ampel_tr,
             filename="ampel_tr.png",
             path=figure_output_dir,
             scale=1.1,
             width=15,
             height=30,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  
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
  tar_group_by(
    data_avgaccs_exp, 
    data_avgaccs, 
    experiment 
  ),  
  
  ## Simulate if desired model terms lead to inflation Type1 error
  tar_file(
    julia_z_script,
    "/Users/roman/GitHub/m4d/julia/Type1error_random_slopes.jl"
  ),
  tar_target(
    simulation_fpr,
    command = {
      plot_file = paste0(figure_output_dir, "ffx_z_values.png")
      dummy = system2(command = julia_executable, args = c(julia_z_script, plot_file),
                      wait=TRUE,# wait for process to be finished before continuing in R
                      stdout = FALSE) # capture output (doesnt work)
      plot_file # TODO, slight errors might not lead to aborting the pipeline
    },
    format = "file"
  ),
  
  ## LMM: is it better to include interactions?
  tar_target(
    r2aic_table,
    rjulia_r2(bind_rows(data_eegnet, data_tsum)) 
  ),
  tar_target(
    r2aic_plot,
    { r2aic_table %>% filter(metric %in% c("R2", "AIC")) %>%
        # capitalize interactions
        mutate(interactions = ifelse(interactions == "false", "Absent", interactions)) %>%
        mutate(interactions = ifelse(interactions == "true", "Present", interactions)) %>%
        ggplot(aes(y=value, x=experiment, fill=interactions)) +
        geom_bar(stat = "identity", position="dodge") +
        scale_fill_grey(start=0.2, end=0.6) +
        facet_wrap(metric ~ model, scales = "free_y",
                   labeller = labeller(metric = label_both)) +
        labs(y="", x="Experiment", fill="Interactions")
    }
  ),
  tar_target(
    name = r2aic_plot_file,
    command = {
      ggsave(plot=r2aic_plot,
             filename="r2aic.png",
             path=figure_output_dir,
             scale=2,
             width=12,
             height=8,
             units="cm",
             dpi=300)
    },
    format="file"),  

  ## HLM: visualize theoretical interactions 
  # TODO, do it instead with significant interactions per exp
  tar_target(
    interaction_chordplot_prior,
    chord_plot(paste0(figure_output_dir,"chord_interactions.png")),
    format = "file"
  ),

  ## LM for sliding
  tar_target(
    name = sliding_LMi2,
    command=lm(formula="tsum ~ (emc + mac + lpf + hpf + ref + det + base + ar) ^ 2", # ^2 includes only 2-way interactions
               data = data_tsum_exp),
    pattern = map(data_tsum_exp),
    iteration = "list"
  ),
  
  ## Estimated marginal means
  
  tar_target(
    name=eegnet_HLM_emm,
    command=est_emm(eegnet_HLMi2,
                    data_eegnet_exp),
    pattern = map(eegnet_HLMi2, data_eegnet_exp),
    iteration = "list"
  ),
  # split means and contrasts
  tar_target(eegnet_HLM_emm_means, eegnet_HLM_emm[[1]], pattern=map(eegnet_HLM_emm)), 
  tar_target(eegnet_HLM_emm_contrasts, eegnet_HLM_emm[[2]], pattern=map(eegnet_HLM_emm)), 
  tar_target(eegnet_HLM_emm_omni, eegnet_HLM_emm[[3]], pattern=map(eegnet_HLM_emm)),

  ### EEGNET interaction
  tar_target(
    name=eegnet_HLMi2_emm,
    command=est_emm_int(eegnet_HLMi2,
                        data_eegnet_exp),
    pattern = map(eegnet_HLMi2, data_eegnet_exp),
    iteration = "list"
  ),
  tar_target(eegnet_HLMi2_emm_means, eegnet_HLMi2_emm[[1]], pattern=map(eegnet_HLMi2_emm)), 
  tar_target(eegnet_HLMi2_emm_contrasts, eegnet_HLMi2_emm[[2]], pattern=map(eegnet_HLMi2_emm)),
  # TODO F values?
  
  ### SLIDING
  tar_target(
    name=sliding_LM_emm,
    command=est_emm(sliding_LMi2, 
                    data_tsum_exp),
    pattern = map(sliding_LMi2, data_tsum_exp),
    iteration = "list"
  ),
  tar_target(sliding_LM_emm_means, sliding_LM_emm[[1]], pattern=map(sliding_LM_emm)), #, iteration="list"
  tar_target(sliding_LM_emm_contrasts, sliding_LM_emm[[2]], pattern=map(sliding_LM_emm)),
  tar_target(sliding_LM_emm_omni, sliding_LM_emm[[3]], pattern=map(sliding_LM_emm)),

  ### Sliding interaction
  tar_target(
    name=sliding_LMi2_emm,
    command=est_emm_int(sliding_LMi2,
                    data_tsum_exp),
    pattern = map(sliding_LMi2, data_tsum_exp),
    iteration = "list"
  ),
  tar_target(sliding_LMi2_emm_means, sliding_LMi2_emm[[1]], pattern=map(sliding_LMi2_emm)), #, iteration="list"
  tar_target(sliding_LMi2_emm_contrasts, sliding_LMi2_emm[[2]], pattern=map(sliding_LMi2_emm)),
  # TODO F values?
  
  ### Sliding Avgaccs (Single)
  tar_target(
    name=sliding_HLMi2_emm,
    command=est_emm(sliding_HLMi2,
                    data_avgaccs_exp),
    pattern = map(sliding_HLMi2, data_avgaccs_exp),
    iteration = "list"
  ),
  tar_target(sliding_HLMi2_emm_means, sliding_HLMi2_emm[[1]], pattern=map(sliding_HLMi2_emm)), #, iteration="list"
  tar_target(sliding_HLMi2_emm_contrasts, sliding_HLMi2_emm[[2]], pattern=map(sliding_HLMi2_emm)),

  
  ## heatmaps of EMMs
  # TODO: use the omni test significances to highlight the facets
  tar_target(eegnet_heatmap,
            command=heatmap(eegnet_HLM_emm_means)),
  tar_target(sliding_heatmap,
             heatmap(sliding_LM_emm_means)),
  tar_target(slidingavgaccs_heatmap,
             heatmap(sliding_HLMi2_emm_means)),
  tar_target(
    name = heatmaps,
    command = {
      ggarrange(eegnet_heatmap + labs(title="EEGNet"), 
                sliding_heatmap + labs(title="Time-resolved"), 
                labels = c("A", "B"),
                ncol = 1, nrow = 2)
    }),
  tar_target(
    name = heatmaps_avgacc,
    command = {
      ggarrange(eegnet_heatmap + labs(title="EEGNet"),
                slidingavgaccs_heatmap + labs(title="Time-resolved"),
                labels = c("A", "B"),
                ncol = 1, nrow = 2)
    }),

  ## interaction plots of EMMs
  tar_target(eegnet_interaction,
            command=interaction_plot(eegnet_HLMi2_emm_means, "EEGNet - "),
            pattern=map(eegnet_HLMi2_emm_means),
            iteration="list"),
  tar_target(sliding_interaction,
             command=interaction_plot(sliding_LMi2_emm_means, "Time-resolved - "),
             pattern = map(sliding_LMi2_emm_means),
             iteration="list"),
  
  ## concatenate all models and data in one target
  tar_target( 
    name=models_combined,
    command=c(eegnet_HLMi2, sliding_LMi2) # TODO add new interaction models
  ), 
  
  ## diagnostics for all models (HLM, LM and experiment wise)
  ### convergence check
  tar_target( 
    name=convergence_checks,
    command=check_convergence(models_combined),
    pattern=map(models_combined),
    iteration="list"
  ), 
  
  ### qq plots
  #### EEGNet
  tar_target(eegnet_HLM_qq,
    command = qqplot(model=eegnet_HLMi2, data=data_eegnet_exp),
    pattern = map(eegnet_HLMi2, data_eegnet_exp),
    iteration ="list"),
  tar_target(eegnet_HLM_qq_comb,
    {plt <- ggarrange(plotlist = eegnet_HLM_qq)
     annotate_figure(plt, top = text_grob("Quantile-quantile plots - EEGNet", 
                     color = "black", face = "bold", size = 16))}),
#  tar_target(eegnet_HLM_qq_comb,
#             {plt <- ggarrange(plotlist = eegnet_HLM_qq_agg)
#             annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - EEGNet", 
#                                                  color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_qq,
    qqplot(model=sliding_LMi2, data=data_tsum_exp),
    pattern = map(sliding_LMi2, data_tsum_exp),
    iteration ="list"),
  tar_target(sliding_LM_qq_comb,
    {plt <- ggarrange(plotlist = sliding_LM_qq)
    annotate_figure(plt, top = text_grob("Quantile-quantile plots - time-resolved", 
                    color = "black", face = "bold", size = 16))}),
  
  ### res_vs_fitted plots
  #### EEGNet
  tar_target(eegnet_HLM_rvf,
             command = rvfplot(model=eegnet_HLMi2, data=data_eegnet_exp),
             pattern = map(eegnet_HLMi2, data_eegnet_exp),
             iteration ="list"),
  tar_target(eegnet_HLM_rvf_comb,
             {plt <- ggarrange(plotlist = eegnet_HLM_rvf)
             annotate_figure(plt, top = text_grob("Residual vs. fitted plots - EEGNet", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_rvf,
             rvfplot(model=sliding_LMi2, data=data_tsum_exp),
             pattern = map(sliding_LMi2, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LM_rvf_comb,
             {plt <- ggarrange(plotlist = sliding_LM_rvf)
             annotate_figure(plt, top = text_grob("Residual vs. fitted plots - time-resolved", 
                             color = "black", face = "bold", size = 16))}),
  
  ### sqrt abs std res_vs_fitted plots
  #### EEGNet
  tar_target(eegnet_HLM_sasrvf,
             command = sasrvfplot(model=eegnet_HLMi2, data=data_eegnet_exp),
             pattern = map(eegnet_HLMi2, data_eegnet_exp),
             iteration ="list"),
  tar_target(eegnet_HLM_sasrvf_comb,
             {plt <- ggarrange(plotlist = eegnet_HLM_sasrvf)
             annotate_figure(plt, top = text_grob("Scale-location-plots - EEGNet", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_sasrvf,
             sasrvfplot(model=sliding_LMi2, data=data_tsum_exp),
             pattern = map(sliding_LMi2, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LM_sasrvf_comb,
             {plt <- ggarrange(plotlist = sliding_LM_sasrvf)
             annotate_figure(plt, top = text_grob("Scale-location-plots - time-resolved", 
                             color = "black", face = "bold", size = 16))}),
  
  
  ## RFX Boxplots
  tar_target(eegnet_RFX,
             rfx_vis(eegnet_HLMi2, data_eegnet_exp),
             pattern=map(eegnet_HLMi2, data_eegnet_exp),
             iteration="list"),
  tar_target(eegnet_RFX_plot,
             {
               plt <- ggarrange(plotlist = eegnet_RFX) #
               annotate_figure(plt, top = text_grob("Random effects - EEGNet", 
                               color = "black", face = "bold", size = 16))
             }),

  ## RFX Intercepts and Participant Demographics
  tar_target(eegnet_rfx_demographics,
             plot_rfx_demographics(eegnet_HLMi2, demographics, data_eegnet_exp, "EEGNet"),
             pattern = map(eegnet_HLMi2, data_eegnet_exp), #, demographics
             iteration = "list"
             ),
  tar_target(eegnet_rfx_demographics_all,
             {ggarrange(plotlist = eegnet_rfx_demographics, ncol=1) %>% 
                annotate_figure(top = text_grob("EEGNet", 
                                                color = "black", face = "bold", size = 16))
               }
  ),
  tar_target(tr_rfx_demographics,
             plot_rfx_demographics(sliding_HLMi2, demographics, data_avgaccs_exp, "Time-resolved"),
             pattern = map(sliding_HLMi2, data_avgaccs_exp), #, demographics
             iteration = "list"
  ),
  tar_target(tr_rfx_demographics_all,
             {ggarrange(plotlist = tr_rfx_demographics, ncol=1) %>% 
               annotate_figure(top = text_grob("Time-resolved", 
                                                color = "black", face = "bold", size = 16))
             }
  ),

  ## RFX Intercepts per Experiment
  tar_target(rfx_eegnet,
             extract_rfx_exp(eegnet_HLMi2, data_eegnet_exp),
             pattern=map(eegnet_HLMi2, data_eegnet_exp),
             # THIS AUTOMATICALLY rbinds, if we don't use iteration="list"
             ),
  tar_target(rfx_eegnet_pairsplot,
             {
               wide_data <- rfx_eegnet %>% 
                 pivot_wider(names_from = Experiment, values_from = "Intercept") %>% 
                 select(-c("Subject")) # remove sub for now
               
               ggpairs(wide_data,
                       upper = list(continuous = wrap(cor_with_p_adjust))) + # BH multiple comparison correction across all comparisons
                 #labs(title="Random Intercept Correlation Between Experiments") +
                 theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                 labs(title="EEGNet")
               
             }
  ),   
  tar_target(rfx_tr,
             extract_rfx_exp(sliding_HLMi2, data_avgaccs_exp),
             pattern=map(sliding_HLMi2, data_avgaccs_exp),
             # THIS AUTOMATICALLY rbinds, if we don't use iteration="list"
  ),
  tar_target(rfx_tr_pairsplot,
             {
               wide_data <- rfx_tr %>% 
                 pivot_wider(names_from = Experiment, values_from = "Intercept") %>% 
                 select(-c("Subject")) # remove sub for now
               
               ggpairs(wide_data,
                       upper = list(continuous = wrap(cor_with_p_adjust))) + # BH multiple comparison correction across all comparisons
                 #labs(title="Random Intercept Correlation Between Experiments") +
                 theme(axis.text.x = element_text(angle = 90, hjust = 1))  +
                 labs(title="Time-resolved")
             }
  ),   


  # TODO: all plot file tarets to the respective creation target, no need to separate
  
  # time resolved all forking paths visualization
  tar_target(tr_accuracy_plot,
             cluster_heatmap(data_sliding, data_tsum)
  ),
  tar_target(
    name = tr_accuracy_plot_file,
    command = {
      ggsave(plot=tr_accuracy_plot,
             filename="tr_accuracy_plot.png",
             path=figure_output_dir,
             scale=1,
             width=20,
             height=20,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  

  # baseline artifact investigation
  tar_target(tr_baseline_artifact_plot,
             plot_baseline_artifacts(data_sliding, data_tsum)
  ),
  tar_target(
    name = tr_baseline_artifact_plot_file,
    command = {
      ggsave(plot=tr_baseline_artifact_plot,
             filename="tr_baseline_artifact_plot.png",
             path=figure_output_dir,
             scale=1.5,
             width=15,
             height=15,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  

  ## Exports for Paper

  ### Figures
  
  tar_target(
    name = overview_file,
    command = {
      ggsave(plot=overview,
             filename="overview.png",
             path=figure_output_dir,
             scale=1,
             width=20,
             height=8,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  tar_target(
    name = overview_poster_file,
    command = {
      ggsave(plot=overview_poster,
             filename="overview_poster.png",
             path=figure_output_dir,
             scale=1,
             width=12,
             height=16,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  tar_target(
    name = overview_eegnet_subjects_file,
    command = {
      ggsave(plot=overview_accuracy,
             filename="overview_eegnet_subjects.png",
             path=figure_output_dir,
             scale=1,
             width=12,
             height=9,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  tar_target(
    name = overview_sliding_avgaccs_file,
    command = {
      ggsave(plot=overview_avgaccs_avgsub,
             filename="overview_sliding_avgaccs.png",
             path=figure_output_dir,
             scale=1,
             width=12,
             height=9,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  

  tar_target(
    name =heatmaps_file,
    command = {
      ggsave(plot=heatmaps,
             filename="heatmaps.png",
             path=figure_output_dir,
             scale=1.5,
             width=22,
             height=12,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  
  tar_target(
    name = heatmap_avgacc_file,
    command = {
      thisPlot <- slidingavgaccs_heatmap + labs(title="Time-resolved")
      ggsave(plot=thisPlot,
             filename="heatmap_avgacc.png",
             path=figure_output_dir,
             scale=1.5,
             width=22,
             height=7,
             units="cm",
             dpi=300)
    },
    format="file"
  ),  

  tar_target(
    name =heatmaps_file_poster,
    command = {
      ggsave(plot=heatmaps_avgacc,
             filename="heatmaps_avgacc.png",
             path=figure_output_dir,
             scale=1.5,
             width=22,
             height=12,
             units="cm",
             dpi=300)
    },
    format="file"
  ),

  # 1 file per branch
  tar_target(eegnet_interaction_filenames, paste0("interactions_eegnet_", experiments, ".png")),
  tar_target(eegnet_interaction_files,
             command = {
               ggsave(plot=eegnet_interaction,
                      filename=eegnet_interaction_filenames,
                      path=figure_output_dir,
                      scale=1.5,
                      width=17,
                      height=17,
                      units="cm",
                      dpi=300)
             },
             pattern=map(eegnet_interaction, eegnet_interaction_filenames),
             format="file"
  ),
  tar_target(sliding_interaction_filenames, paste0("interactions_sliding_", experiments, ".png")),
  tar_target(sliding_interaction_files,
             command = {
               ggsave(plot=sliding_interaction,
                      filename=sliding_interaction_filenames,
                      path=figure_output_dir,
                      scale=1.5,
                      width=17,
                      height=17,
                      units="cm",
                      dpi=300)
             },
             pattern=map(sliding_interaction, sliding_interaction_filenames),
             format="file"
  ),

  
  tar_target(
    name = eegnet_RFX_plot_file,
    command = {
      ggsave(plot=eegnet_RFX_plot,
             filename="RFX.png",
             path=figure_output_dir,
             scale=2,
             width=15,
             height=15,
             units="cm",
             dpi=300)
    },
    format="file"),

  tar_target(
    name = rfx_eegnet_pairs_file,
    command = {
      ggsave(plot=rfx_eegnet_pairsplot,
             filename="rfx_eegnet_pairs.png",
             path=figure_output_dir,
             scale=1,
             width=15,
             height=15,
             units="cm",
             dpi=300)
    }),
    
    tar_target(
      name = rfx_tr_pairs_file,
      command = {
        ggsave(plot=rfx_tr_pairsplot,
               filename="rfx_tr_pairs.png",
               path=figure_output_dir,
               scale=1,
               width=15,
               height=15,
               units="cm",
               dpi=300)
      },
  format="file"),
  
  tar_target(
    name = eegnet_rfx_demographics_file,
    command = {
      ggsave(plot=eegnet_rfx_demographics_all,
             filename="eegnet_rfx_demographics.png",
             path=figure_output_dir,
             scale=1.5,
             width=15,
             height=20,
             units="cm",
             dpi=300)
    },
    format="file"),
  tar_target(
    name = tr_rfx_demographics_file,
    command = {
      ggsave(plot=tr_rfx_demographics_all,
             filename="tr_rfx_demographics.png",
             path=figure_output_dir,
             scale=1.5,
             width=15,
             height=20,
             units="cm",
             dpi=300)
    },
    format="file"),



  ### Tables
  tar_target(
    name = table_f_eegnet,
    command = output.table.f(eegnet_HLM_emm_omni,
                             filename=paste0(table_output_dir, "eegnet_omni.tex"),
                             thisLabel = "eegnet_omni",
                             thisCaption = "Significant effects of preprocessing on EEGNet decoding performance, separately for each experiment. F-tests were performed for each processing step. Stars indicate the signicifance level ('.'~$p<0.1$; '*'~$p<0.05$; '**'~$p<0.01$; '***'~$p<0.001$), false discovery rate-corrected using the Benjamini–Yekutieli procedure. \\textit{Ocular}: ocular artifact correction; \\textit{muscle}: muscle artifact correction; \\textit{ICA}: independent component analysis, \\textit{low pass}: low pass filter; \\textit{high pass}: high pass filter."
                             ),
    format = "file"
  ),
  tar_target(
    name = table_f_sliding,
    command = output.table.f(sliding_LM_emm_omni,
                             filename=paste0(table_output_dir, "sliding_omni.tex"),
                             thisLabel = "sliding_omni",
                             thisCaption = "Significant effects of preprocessing on time-resolved decoding performance, separately for each experiment. See Table~\\ref{eegnet_omni} for details."
    ),
    format = "file"
  ),

  tar_target(
    name = table_con_eegnet,
    command = output.table.con(eegnet_HLM_emm_contrasts,
                             filename=paste0(table_output_dir, "eegnet_contrasts.tex"),
                             thisLabel = "eegnet_contrasts",
                             thisCaption = "For each experiment, pairwise post-hoc comparisons in EEGNet decoding performance within each preprocessing step using Tukey adjustment. See Table~\\ref{eegnet_omni} for details."
    ),
    format = "file"
  ),
  tar_target(
    name = table_con_sliding,
    command = output.table.con(sliding_LM_emm_contrasts,
                             filename=paste0(table_output_dir, "sliding_contrasts.tex"),
                             thisLabel = "sliding_contrasts",
                             thisCaption = "For each experiment, pairwise post-hoc comparisons in time-resolved decoding performance within each preprocessing step using Tukey adjustment. See Table~\\ref{eegnet_omni} for details."
    ),
    format = "file"
  ),

  #######################################################################
  ########### ALTERNATIVE PIPELINE ######################################
  #######################################################################


# alternative order of the pipeline (appendix)
tar_target(
  name = eegnet_file_ALT,
  command = "../models/eegnet_original_order.csv", # TODO: put this into a different folder
  format = "file"
),
tar_target(
  name = sliding_file_ALT,
  command = "../models/sliding_original_order.csv",
  format = "file"
),
tar_target(
  name = tsum_file_ALT,
  command = "../models/sliding_tsums_original_order.csv",
  format = "file"
),
# load models
tar_target(
  name = jLMM_file_ERN_ALT,
  command = '../julia/alternative_order/model_ERN_eegnet.rds',
  format = "file_fast" # file_fast checks if the file is up to date!!
),
tar_target(
  name = jLMM_file_LRP_ALT,
  command = '../julia/alternative_order/model_LRP_eegnet.rds',
  format = "file_fast"
),
tar_target(
  name = jLMM_file_MMN_ALT,
  command = '../julia/alternative_order/model_MMN_eegnet.rds',
  format = "file_fast"
),
tar_target(
  name = jLMM_file_N170_ALT,
  command = '../julia/alternative_order/model_N170_eegnet.rds',
  format = "file_fast"
),
tar_target(
  name = jLMM_file_N2pc_ALT,
  command = '../julia/alternative_order/model_N2pc_eegnet.rds',
  format = "file_fast"
),
tar_target(
  name = jLMM_file_N400_ALT,
  command = '../julia/alternative_order/model_N400_eegnet.rds',
  format = "file_fast"
),
tar_target(
  name = jLMM_file_P3_ALT,
  command = '../julia/alternative_order/model_P3_eegnet.rds',
  format = "file_fast"
),
tar_target(
  name = jLMM_files_ALT,
  command = c(jLMM_file_ERN_ALT, jLMM_file_LRP_ALT, jLMM_file_MMN_ALT, jLMM_file_N170_ALT, jLMM_file_N2pc_ALT, jLMM_file_N400_ALT, jLMM_file_P3_ALT),
),
tar_target(name = eegnet_HLMi2_ALT, 
           readRDS(jLMM_files_ALT),
           pattern=jLMM_files_ALT,
           iteration="list"),

# TODO: recoding is probably adjusted to new pipeline, check if functions should be copied and adjusted or only adjusted
## import and recode datasets
tar_target(
  name = data_eegnet_ALT,
  command = {get_preprocess_data_ALT(eegnet_file_ALT)} #  %>% filter(dataset == "ERPCORE") %>% select(-c(dataset))
),
tar_target(
  name = data_sliding_ALT,
  command = {get_preprocess_data_ALT(sliding_file_ALT) %>% select(-c(forking_path))} # , dataset  %>% filter(dataset == "ERPCORE") 
),
tar_target(
  name = data_tsum_ALT,
  command = {get_preprocess_data_ALT(tsum_file_ALT) %>% select(-c(forking_path))} # , dataset  %>% filter(dataset == "ERPCORE") 
),

## Overview of decoding accuracies for each pipeline
tar_target(
  name = overview_accuracy_ALT,
  command = raincloud_acc(data_eegnet_ALT, title = "EEGNet")
),
tar_target( # average across subjects for each pipeline
  name = overview_accuracy_avgsub_ALT,
  command = raincloud_acc(data_eegnet_ALT %>%
                            group_by(emc, mac, lpf, hpf, ref, det, base, ar, experiment) %>% #ref, hpf, lpf, emc, mac, det, base, ar, experiment
                            summarize(accuracy = mean(accuracy)) %>% 
                            select(accuracy, everything()), # put the accuracy in the first column
                          title = "EEGNet")
),
tar_target(
  name = overview_tsum_ALT,
  command = raincloud_acc(data_tsum_ALT, title = "Time-resolved")
),
tar_target(
  name = overview_ALT,
  command = {
    ggarrange(overview_accuracy_avgsub_ALT, overview_tsum_ALT, 
              labels = c("A", "B"),
              ncol = 2, nrow = 1)
  }
), 

## GROUPINGS
tar_group_by(
  data_eegnet_exp_ALT, 
  data_eegnet_ALT, 
  experiment,
),    
tar_group_by(
  data_tsum_exp_ALT, 
  data_tsum_ALT, 
  experiment 
),    

## LM for sliding
tar_target(
  name = sliding_LMi2_ALT,
  command=lm(formula="tsum ~ (emc + mac + lpf + hpf + ref + det + base + ar) ^ 2", # ^2 includes only 2-way interactions
             data = data_tsum_exp_ALT),
  pattern = map(data_tsum_exp_ALT),
  iteration = "list"
),

## Estimated marginal means

tar_target(
  name=eegnet_HLM_emm_ALT,
  command=est_emm(eegnet_HLMi2_ALT,
                  data_eegnet_exp_ALT),
  pattern = map(eegnet_HLMi2_ALT, data_eegnet_exp_ALT),
  iteration = "list"
),
# split means and contrasts
tar_target(eegnet_HLM_emm_means_ALT, eegnet_HLM_emm_ALT[[1]], pattern=map(eegnet_HLM_emm_ALT)), 
tar_target(eegnet_HLM_emm_contrasts_ALT, eegnet_HLM_emm_ALT[[2]], pattern=map(eegnet_HLM_emm_ALT)), 
tar_target(eegnet_HLM_emm_omni_ALT, eegnet_HLM_emm_ALT[[3]], pattern=map(eegnet_HLM_emm_ALT)),

### SLIDING
tar_target(
  name=sliding_LM_emm_ALT,
  command=est_emm(sliding_LMi2_ALT, 
                  data_tsum_exp_ALT),
  pattern = map(sliding_LMi2_ALT, data_tsum_exp_ALT),
  iteration = "list"
),
tar_target(sliding_LM_emm_means_ALT, sliding_LM_emm_ALT[[1]], pattern=map(sliding_LM_emm_ALT)), #, iteration="list"
tar_target(sliding_LM_emm_contrasts_ALT, sliding_LM_emm_ALT[[2]], pattern=map(sliding_LM_emm_ALT)),
tar_target(sliding_LM_emm_omni_ALT, sliding_LM_emm_ALT[[3]], pattern=map(sliding_LM_emm_ALT)),

# HEATMAP
tar_target(eegnet_heatmap_ALT,
           command=heatmap_ALT(eegnet_HLM_emm_means_ALT)),
tar_target(sliding_heatmap_ALT,
           heatmap_ALT(sliding_LM_emm_means_ALT)),
tar_target(
  name = heatmaps_ALT,
  command = {
    ggarrange(eegnet_heatmap_ALT + labs(title="EEGNet"), 
              sliding_heatmap_ALT + labs(title="Time-resolved"), 
              labels = c("A", "B"),
              ncol = 1, nrow = 2)
  }),

tar_target(
  name = overview_file_ALT,
  command = {
    ggsave(plot=overview_ALT,
           filename="overview_ALT.png",
           path=figure_output_dir,
           scale=1,
           width=20,
           height=8,
           units="cm",
           dpi=300)
  },
  format="file"
),  
tar_target(
  name =heatmaps_file_ALT,
  command = {
    ggsave(plot=heatmaps_ALT,
           filename="heatmaps_ALT.png",
           path=figure_output_dir,
           scale=1.5,
           width=22,
           height=12,
           units="cm",
           dpi=300)
  },
  format="file"
)  

)

