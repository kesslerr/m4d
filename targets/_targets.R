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
               "xtable" # export R tables to latex files
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
  #   c("EEGNET","Time-Resolved")
  # ),
  # 
  
  ## define raw data files
  tar_target(
    name = eegnet_file,
    command = "eegnet.csv", # TODO: put this into a different folder
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
  tar_target(
    name = demographics_file,
    command = '../data/erpcore/participants.tsv',
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
  tar_target(
    name = demographics,
    command = {read_tsv(demographics_file) %>% mutate_if(is.character, as.factor)}
  ),
  
  ## Example results of Luck forking path
  tar_target(
    name = timeresolved_luck,
    command = timeresolved_plot(data_sliding)
  ),
  
  ## Overview of decoding accuracies for each pipeline
  tar_target(
    name = overview_accuracy,
    command = raincloud_acc(data_eegnet, title = "EEGNet")
  ),
  tar_target( # average across subjects for each pipeline
    name = overview_accuracy_avgsub,
    command = raincloud_acc(data_eegnet %>%
                              group_by(ref, hpf, lpf, emc, mac, base, det, ar, experiment) %>%
                              summarize(accuracy = mean(accuracy)) %>% 
                              select(accuracy, everything()), # put the accuracy in the first column
                            title = "EEGNet")
  ),
  tar_target(
    name = overview_tsum,
    command = raincloud_acc(data_tsum, title = "Time-Resolved")
  ),
  tar_target(
    name = overview,
    command = {
      ggarrange(overview_accuracy_avgsub, overview_tsum, 
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
  
  
  ## Simulate if desired model terms lead to inflation of false positive rate
  tar_file(
   julia_simulation_fpr_script,
   "/Users/roman/GitHub/m4d/julia/simulation_rfxslopes_fp.jl"
  ),
  tar_target(
    simulation_fpr,
    command = {
      plot_file = paste0(figure_output_dir, "simulations_rfxslopes_fp.png")
      dummy = system2(command = julia_executable, args = c(julia_simulation_fpr_script, plot_file),
              wait=TRUE,# wait for process to be finished before continuing in R
              stdout = TRUE) # capture output (doesnt work)
      plot_file # TODO, slight errors might not lead to aborting the pipeline
    },
    format = "file"
  ),
  
  ## HLM
  tar_target(
    eegnet_HLM,
    rjulia_mlm(data_eegnet_exp, interactions=FALSE),
    pattern = map(data_eegnet_exp),
    iteration = "list"
  ),
  tar_target(
   eegnet_HLMi2,
   rjulia_mlm(data_eegnet_exp, interactions=TRUE),
   pattern = map(data_eegnet_exp),
   iteration = "list"
  ),
  
  tar_target(
    r2aic_table,
    rjulia_r2(bind_rows(data_eegnet, data_tsum)) # bind_rows does completes with NA in case of mismatching col names
  ),
  tar_target(
    r2aic_plot,
    { r2aic_table %>% filter(metric %in% c("R2", "AIC")) %>%
        ggplot(aes(y=value, x=experiment, fill=interactions)) +
        geom_bar(stat = "identity", position="dodge") +
        scale_fill_grey(start=0.2, end=0.6) +
        facet_wrap(metric ~ model, scales = "free_y") +
        labs(y="")
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
             dpi=500)
    },
    format="file"),  
  
  # tar_target(
  #   eegnet_HLMi2,
  #   rjulia_mlm(data_eegnet_exp, interactions=TRUE),
  #   pattern = map(data_eegnet_exp),
  #   iteration = "list"
  # ),  
  
  tar_target(
    interaction_chordplot_prior,
    chord_plot(paste0(figure_output_dir,"chord_interactions.png")),
    format = "file"
  ),

  #tar_target(
  #  name = eegnet_HLM,
  #  command=lme4::lmer(formula="accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)",
  #                     control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
  #                     data = data_eegnet_exp),
  #  pattern = map(data_eegnet_exp),
  #  iteration = "list"
  #),  

  ## LM for sliding
  tar_target(
    name = sliding_LM,
    command=lm(formula="tsum ~ ref + hpf + lpf + emc + mac + base + det + ar",
               data = data_tsum_exp),
    pattern = map(data_tsum_exp),
    iteration = "list"
  ),
  
  ### TODO: TEST: Sliding with all interactions
  tar_target(
    name = sliding_LMi2,
    command=lm(formula="tsum ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2", # ^2 includes only 2-way interactions
               data = data_tsum_exp),
    pattern = map(data_tsum_exp),
    iteration = "list"
  ),
  tar_target(
    name = sliding_LMi3,
    command=lm(formula="tsum ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 3", 
               data = data_tsum_exp),
    pattern = map(data_tsum_exp),
    iteration = "list"
  ),  
  
  ## Estimated marginal means
  
  # TODO: var explained of random slopes and random subject intercepts?
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

  # TODO EMM interactions
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
    command=est_emm(sliding_LM, 
                    data_tsum_exp),
    pattern = map(sliding_LM, data_tsum_exp),
    iteration = "list"
  ),
  tar_target(sliding_LM_emm_means, sliding_LM_emm[[1]], pattern=map(sliding_LM_emm)), #, iteration="list"
  tar_target(sliding_LM_emm_contrasts, sliding_LM_emm[[2]], pattern=map(sliding_LM_emm)),
  tar_target(sliding_LM_emm_omni, sliding_LM_emm[[3]], pattern=map(sliding_LM_emm)),

  # TODO Omni, also other models, and maybe combine it with the targets using dyn branching?
  
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
  
  
  ## heatmaps of EMMs
  # TODO: use the omni test significances to highlight the facets
  tar_target(eegnet_heatmap,
            command=heatmap(eegnet_HLM_emm_means)),
  tar_target(sliding_heatmap,
             heatmap(sliding_LM_emm_means)),
  tar_target(
    name = heatmaps,
    command = {
      ggarrange(eegnet_heatmap + labs(title="EEGNet"), 
                sliding_heatmap + labs(title="Time-resolved"), 
                labels = c("A", "B"),
                ncol = 1, nrow = 2)
    }),
  # TODO: main effects with only means?
  
  ## interaction plots of EMMs
  tar_target(eegnet_interaction,
            command=interaction_plot(eegnet_HLMi2_emm_means, "EEGNet - "),
            pattern=map(eegnet_HLMi2_emm_means),
            iteration="list"),
  tar_target(sliding_interaction,
             command=interaction_plot(sliding_LMi2_emm_means, "Time-resolved - "),
             pattern = map(sliding_LMi2_emm_means),
             iteration="list"),
  
  ## concatenate all models and datas in one target
  tar_target( 
    name=models_combined,
    command=c(eegnet_HLM, eegnet_HLMi2, sliding_LM, sliding_LMi2) # TODO add new interaction models
  ), 
  
  ## diagnostics for all models (HLM, LM, ALL and experiment wise)
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
     annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - EEGNet", 
                     color = "black", face = "bold", size = 16))}),
#  tar_target(eegnet_HLM_qq_comb,
#             {plt <- ggarrange(plotlist = eegnet_HLM_qq_agg)
#             annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - EEGNet", 
#                                                  color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_qq,
    qqplot(model=sliding_LM, data=data_tsum_exp),
    pattern = map(sliding_LM, data_tsum_exp),
    iteration ="list"),
  tar_target(sliding_LM_qq_comb,
    {plt <- ggarrange(plotlist = sliding_LM_qq)
    annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - Time-Resolved", 
                    color = "black", face = "bold", size = 16))}),
  
  #### SLIDING WITH INTERACTIONS
  tar_target(sliding_LMi2_qq,
             qqplot(model=sliding_LMi2, data=data_tsum_exp),
             pattern = map(sliding_LMi2, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LMi2_qq_comb,
             {plt <- ggarrange(plotlist = sliding_LMi2_qq)
              annotate_figure(plt, top = text_grob("Quantile-Quantile Plots - Time-Resolved", 
                                                  color = "black", face = "bold", size = 16))}),
  
  ### res_vs_fitted plots
  #### EEGNet
  tar_target(eegnet_HLM_rvf,
             command = rvfplot(model=eegnet_HLMi2, data=data_eegnet_exp),
             pattern = map(eegnet_HLMi2, data_eegnet_exp),
             iteration ="list"),
  tar_target(eegnet_HLM_rvf_comb,
             {plt <- ggarrange(plotlist = eegnet_HLM_rvf)
             annotate_figure(plt, top = text_grob("Residual vs. Fitted Plots - EEGNet", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_rvf,
             rvfplot(model=sliding_LM, data=data_tsum_exp),
             pattern = map(sliding_LM, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LM_rvf_comb,
             {plt <- ggarrange(plotlist = sliding_LM_rvf)
             annotate_figure(plt, top = text_grob("Residual vs. Fitted Plots - Time-Resolved", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING WITH INTERACTIONS
  tar_target(sliding_LMi2_rvf,
             rvfplot(model=sliding_LMi2, data=data_tsum_exp),
             pattern = map(sliding_LMi2, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LMi2_rvf_comb,
             {plt <- ggarrange(plotlist = sliding_LMi2_rvf)
             annotate_figure(plt, top = text_grob("Residual vs. Fitted Plots - Time-Resolved", 
                                                  color = "black", face = "bold", size = 16))}),

  ### sqrt abs std res_vs_fitted plots
  #### EEGNet
  tar_target(eegnet_HLM_sasrvf,
             command = sasrvfplot(model=eegnet_HLMi2, data=data_eegnet_exp),
             pattern = map(eegnet_HLMi2, data_eegnet_exp),
             iteration ="list"),
  tar_target(eegnet_HLM_sasrvf_comb,
             {plt <- ggarrange(plotlist = eegnet_HLM_sasrvf)
             annotate_figure(plt, top = text_grob("Scale-Location-Plots - EEGNet", 
                             color = "black", face = "bold", size = 16))}),
  
  #### SLIDING
  tar_target(sliding_LM_sasrvf,
             sasrvfplot(model=sliding_LM, data=data_tsum_exp),
             pattern = map(sliding_LM, data_tsum_exp),
             iteration ="list"),
  tar_target(sliding_LM_sasrvf_comb,
             {plt <- ggarrange(plotlist = sliding_LM_sasrvf)
             annotate_figure(plt, top = text_grob("Scale-Location-Plots - Time-Resolved", 
                             color = "black", face = "bold", size = 16))}),
  
  
  ## RFX Boxplots
  tar_target(eegnet_RFX,
             rfx_vis(eegnet_HLMi2, data_eegnet_exp),
             pattern=map(eegnet_HLMi2, data_eegnet_exp),
             iteration="list"),
  tar_target(eegnet_RFX_plot,
             {
               plt <- ggarrange(plotlist =eegnet_RFX)
               annotate_figure(plt, top = text_grob("Random Effects - EEGNet", 
                               color = "black", face = "bold", size = 16))
             }),

  ## RFX Intercepts and Participant Demographics
  tar_target(rfx_demographics,
             plot_rfx_demographics(eegnet_HLMi2, demographics),
             pattern = map(eegnet_HLMi2) #, demographics
             ),
  
  ## RFX Intercepts per Experiment
  tar_target(rfx,
             extract_rfx_exp(eegnet_HLMi2, data_eegnet_exp),
             pattern=map(eegnet_HLMi2, data_eegnet_exp),
             # THIS AUTOMATICALLY rbinds, if we don't use iteration="list"
             ),
  tar_target(rfx_plot,
             {
               wide_data <- rfx %>% 
                 pivot_wider(names_from = Experiment, values_from = "Intercept") %>% 
                 select(-c("Subject")) # remove sub for now
               
               ggpairs(wide_data) + labs(title="RFX Correlation Between Experiments")
             }
  ),
  
  
  ## Exports for Paper

  ### Figures
  
  tar_target(
    name = overview_file,
    command = {
      ggsave(plot=overview,
             filename="overview.png",
             path=figure_output_dir,
             scale=2,
             width=12,
             height=9,
             units="cm",
             dpi=500)
    },
    format="file"
  ),  
  tar_target(
    name = overview_eegnet_subjects_file,
    command = {
      ggsave(plot=overview_accuracy,
             filename="overview_eegnet_subjects.png",
             path=figure_output_dir,
             scale=2,
             width=12,
             height=5,
             units="cm",
             dpi=500)
    },
    format="file"
  ),  
  
  tar_target(
    name = timeresolved_luck_file,
    command = {
      ggsave(plot=timeresolved_luck,
             filename="timeresolved_luck.png",
             path=figure_output_dir,
             scale=2,
             width=12,
             height=16,
             units="cm",
             dpi=500)
    },
    format="file"
  ),  
  
  tar_target(
    name =heatmaps_file,
    command = {
      ggsave(plot=heatmaps,
             filename="heatmaps.png",
             path=figure_output_dir,
             scale=2,
             width=12,
             height=12,
             units="cm",
             dpi=500)
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
                      scale=2,
                      width=17,
                      height=17,
                      units="cm",
                      dpi=500)
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
                      scale=2,
                      width=17,
                      height=17,
                      units="cm",
                      dpi=500)
             },
             pattern=map(sliding_interaction, sliding_interaction_filenames),
             format="file"
  ),

  #tar_target(
  #  interaction_file_eegnet,
  #  command = {
  #    ggarrange(plotlist = interaction_eegnet, ncol=2) %>% 
  #      annotate_figure(top = text_grob("Interaction Effects - EEGNet", 
  #                                      color = "black", face = "bold", size = 16)) %>% 
  #      ggsave(filename="interaction_eegnet.png",
  #             path=figure_output_dir,
  #             scale=2,
  #             width=12,
  #             height=12,
  #             units="cm",
  #             dpi=500)
  #  }
  #),
  
  tar_target(
    name = eegnet_RFX_plot_file,
    command = {
      ggsave(plot=eegnet_RFX_plot,
             filename="RFX.png",
             path=figure_output_dir,
             scale=2,
             width=12,
             height=12,
             units="cm",
             dpi=500)
    },
    format="file"),

  ### Tables
  tar_target(
    name = table_f_eegnet,
    command = output.table.f(eegnet_HLM_emm_omni,
                             filename=paste0(table_output_dir, "eegnet_omni.tex"),
                             thisLabel = "eegnet_omni",
                             thisCaption = "Significant differences in EEGNet decoding performances within each processing step, separate for each experiments model. ALL corresponds to the combined model including all experiments. F-tests were conducted for each processing step. Stars indicate level of signicifance ('.'~$p<0.1$; '*'~$p<0.05$; '**'~$p<0.01$; '***'~$p<0.001$; '/'~N/A). Significances were FDR corrected using Benjamini–Yekutieli. Correction was applied across all models and processing steps."
                             ),
    format = "file"
  ),
  tar_target(
    name = table_f_sliding,
    command = output.table.f(sliding_LM_emm_omni,
                             filename=paste0(table_output_dir, "sliding_omni.tex"),
                             thisLabel = "sliding_omni",
                             thisCaption = "Significant differences in time-resolved decoding performances within each processing step, separate for each experiments model. See \\ref{eegnet_omni} for details."
    ),
    format = "file"
  ),

  tar_target(
    name = table_con_eegnet,
    command = output.table.con(eegnet_HLM_emm_contrasts,
                             filename=paste0(table_output_dir, "eegnet_contrasts.tex"),
                             thisLabel = "eegnet_contrasts",
                             thisCaption = "Significant differences in EEGNet decoding performances within each processing step, separate for each experiments model. See \\ref{eegnet_omni} for details."
    ),
    format = "file"
  ),
  tar_target(
    name = table_con_sliding,
    command = output.table.con(sliding_LM_emm_contrasts,
                             filename=paste0(table_output_dir, "sliding_contrasts.tex"),
                             thisLabel = "sliding_contrasts",
                             thisCaption = "Significant differences in time-resolved decoding performances within each processing step, separate for each experiments model. See \\ref{eegnet_omni} for details."
    ),
    format = "file"
  )
  
#  tar_target(
#    name=eegnet_HLM_simulations,
#    command=lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + experiment + (1 | subject)",
#                 control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
#                 data = data_eegnet)
#  ),
  
  # TODO: i could synchronize the analyses of eegnet/sliding by just using pattern instead of two targets? (no group by)

# TODO: add another dataset (infants)


)



