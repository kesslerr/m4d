# functions archive



# ecdf plot with the best pipeline(s) marked for each experiment
ecdf <- function(data){
  
  best_data = data.frame()
  for (experiment_val in c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")){
    newdata <- data %>%
      #group_by(ref, hpf, lpf, emc, mac, det, base, ar) %>%
      #summarize(tsum = mean(tsum)) %>%
      filter(experiment==experiment_val) %>%
      mutate(performance = # TODO: adjust to the one or many values which are best according to the statistics you chose
               (ar==FALSE) & 
               #ref=="P9P10"
               #base=="400ms" & 
               (det=="linear") & 
               (emc=="None") & 
               (mac=="None") & 
               (hpf==0.5) & 
               (lpf==6)
      ) %>% # if these conditions are met, then write TRUE, else FALSE
      #ungroup() %>% # because row names etc are not correct in a grouped df
      arrange(tsum) %>% # sort by ascending tsum (if not, write desc(tsum))
      mutate(idx = as.numeric(row.names(.))/1152) # index column
    best_data = rbind(best_data, newdata)
  }
  
  ggplot(best_data, aes(x=tsum)) +
    geom_hline(data =filter(best_data, performance==TRUE),
               aes(yintercept = idx),
               color="darkgrey") +
    stat_ecdf() +
    facet_wrap(experiment ~., scales = "free_x")
}


# # merge MM, contrasts, OMNI
# # DEBUG
# emm <-  tar_read(eegnet_HLM_emm_means_comb)
# con <-  tar_read(eegnet_HLM_emm_contrasts_comb)
# #omn <-  tar_read(eegnet_HLM_emm_omni_comb)
# 
# # combine dataframes
# merged_df <- con %>% 
#   left_join(., emm, by = c("experiment", "variable", "level.1" = "level")) %>% 
#   rename(emmean.1 = emmean) %>%
#   left_join(., emm, by = c("experiment", "variable", "level.2" = "level")) %>% 
#   rename(emmean.2 = emmean)
# 


paired_tests <- function(data, study="ERPCORE"){
  results <- data %>%
    group_by(variable) %>%
    pairwise_t_test(accuracy ~ factor, paired = TRUE, p.adjust.method = "BY",
                    pool.sd = FALSE, detailed = TRUE)
  
  results
}


# Muscle artifact correction components per LPF 
# TODO: this only worked with the old order, not necessary now!
muscle_lpf <- function(ICA="EMG"){
  
  subjects = c("sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008", "sub-009", "sub-010", "sub-011", "sub-012", "sub-013", "sub-014", "sub-015", "sub-016", "sub-017", "sub-018", "sub-019", "sub-020", "sub-021", "sub-022", "sub-023", "sub-024", "sub-025", "sub-026", "sub-027", "sub-028", "sub-029", "sub-030", "sub-031", "sub-032", "sub-033", "sub-034", "sub-035", "sub-036", "sub-037", "sub-038", "sub-039", "sub-040")
  experiments = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")
  
  results = data.frame()
  for (experiment in experiments){
    for (sub in subjects){
      example_char_file <- paste0("/Users/roman/GitHub/m4d/data/interim/",experiment,"/",sub,"/characteristics.json")
      
      # read the file
      df <- jsonlite::fromJSON(example_char_file)
      
      if (ICA == "EMG"){
        unique_emc_pipelines <- names(df$`ICA EMG`)
      } else if (ICA == "EOG"){
        unique_emc_pipelines <- names(df$`ICA EOG`)
      }
      # debug
      #pipeline = unique_emc_pipelines[1]
      
      for (pipeline in unique_emc_pipelines) {
        if (ICA == "EMG"){
          n_comp <- df$`ICA EMG`[[pipeline[[1]][[1]]]]$n_components
        } else if (ICA == "EOG"){
          n_comp <- df$`ICA EOG`[[pipeline[[1]][[1]]]]$n_components
        }
        # split pipeline str at each underscore
        splits <- strsplit(pipeline, "_")[[1]]
        thisResult <- data.frame("ref" = splits[1],
                                 "hpf" = splits[2],
                                 "lpf" = splits[3],
                                 #"emc" = splits[4],
                                 #"mac" = splits[5],
                                 "components" = n_comp,
                                 "experiment" = experiment,
                                 "subject" = sub)
        results <- bind_rows(results, thisResult)
      }
    }
  }
  
  results$lpf <- factor(results$lpf, levels = c("6", "20", "45", "None"))
  library(ggplot2)
  
  # MAC
  p <- ggplot(results, aes(x=lpf, y=components)) + 
    geom_boxplot() +
    labs(y="# of dropped components", x="Low-pass filter (Hz)", title = "Muscle artifact correction via ICA") +
    facet_grid(experiment ~ .)
  
  return(p)
}

estimate_marginal_means_sliding <- function(data, per_exp = FALSE){
  # mean values
  if (per_exp == TRUE){
    experiments <- c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")
  } else {
    experiments <- c("dummy")
  }
  all_results <- data.frame()
  for (experiment_value in experiments){
    for (variable in c("emc","mac","lpf","hpf","ref","det","base","ar",)) { #"ref","hpf","lpf","det","base","ar","emc","mac"
      
      result <- data %>%
        {if(per_exp==TRUE) filter(., experiment == experiment_value) else . } %>%
        #filter(experiment == experiment_value) %>%
        group_by(experiment, !!sym(variable)) %>% # TODO, add experiment as first grouping variable for each 
        summarize(tsum_values = list(tsum)) %>%
        unnest(cols = tsum_values) %>%
        group_by(!!sym(variable)) %>%
        summarize(tsum=mean(tsum_values)) %>%
        pivot_longer(
          cols = c(variable), 
          names_to = "variable",
          values_to = "level"
        ) %>%
        {if(per_exp==TRUE) mutate(., experiment = experiment_value) else . }
      
      all_results <- rbind(all_results, result)
    }
  }
  all_results
}


# concatenate experiment and whole 
combine_single_whole <- function(single, whole){
  whole$experiment = "ALL"
  rbind(whole, single)
}


#DEBUG
#data = tar_read(data_eegnet_exp, branches=1) %>% filter(experiment == "ERN")
rjulia_mlm_with_random_slopes <- function(data, interactions = FALSE){
  # INFO: Julia session stays open once initialized, no current way to restart it, so afex::lmer_alt will replace lme4::lmer as soon as the first case with zerocorr or :: is run.
  julia_library("Parsers, DataFrames, CSV, Plots, MixedModels, RData, CategoricalArrays")
  if (interactions == TRUE){
    julia_command("ENV[\"LMER\"] = \"afex::lmer_alt\"") # set before import RCall and JellyMe4 to be able to convert zerocorr(rfx) correctly; https://github.com/palday/JellyMe4.jl (README)
  } # caution, if zerocorr is used, now julia will automatically use afex::lmer_alt, and from then on, use lmer_alt as reference R object for the remainder of the session
  julia_library("RCall, JellyMe4")
  
  julia_assign("data", data) # bring data into julia
  if (interactions == FALSE){
    julia_command("formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + det + base + ar + (ref + hpf + lpf + emc + mac + det + base + ar | subject));")
  } else {
    julia_command("formula = @formula(accuracy ~ (ref + hpf + lpf) ^ 2 + zerocorr( (ref + hpf + lpf) ^ 2 | subject));") ## TODO all variables
  }
  julia_command("model = fit(LinearMixedModel, formula, data);") 
  julia_command("rmodel = (model, data);") # make it a tuple for conversion (Julia model doesn't have the data saved, but R analogue lmer model does); https://github.com/palday/JellyMe4.jl/issues/51, 
  julia_command("RCall.Const.GlobalEnv[:rmodel] = robject(:lmerMod, rmodel);") # alternative to @rput; https://github.com/palday/JellyMe4.jl/issues/72
  rmodel
}


estimate_marginal_means <- function(data, variables){
  data_list <- list()
  for (variable in variables) {
    average_data <- data %>%
      group_by(subject, !!sym(variable)) %>%
      summarize(accuracy = mean(accuracy))# %>%
    #ungroup()
    average_data$variable <- names(average_data)[2]
    average_data$factor <- average_data[[variable]]
    average_data[[variable]] <- NULL
    data_list <- append(data_list, list(average_data))
  }
  # Concatenate the data frames in the list
  bind_rows(data_list)
}


timeresolved_avg_acc <- function(data, subject_wise = FALSE){
  #data <- tar_read(data_sliding)
  experiments <- c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")
  
  baseline_end = c(-0.2, -0.4, 0., 0., 0., 0., 0.) # TODO write paper: in LRP the baselines end at different timepoints, therefore the decoding windows are different for 200ms and 400ms, Here, I chose to equalize the AvgAccuracy esimation to keep it fair 
  names(baseline_end) <- c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")
  
  new_data = data.frame()
  for (exp in experiments) {
    data_tmp <- data %>% 
      filter(experiment == exp) %>%
      filter(times >= baseline_end[exp])
    avg_accuracy <- data_tmp %>%
      group_by(emc, mac, lpf, hpf, ref, det, base, ar) %>%
      summarize(accuracy = mean(`balanced accuracy`)) %>%
      # reorder columns with accuracy on first place
      select(accuracy, everything()) %>%
      mutate(experiment = exp)
    new_data <- rbind(new_data, avg_accuracy)
  }
  new_data$experiment <- factor(new_data$experiment, levels = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3"))
  new_data
}


# ungroup targets across all branches
ungrouping <- function(input){
  data = data.frame()
  i <- 0
  for (experiment in c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")){
    i <- i + 1
    #tmp <- tar_read(eegnet_HLM_exp_emm_means, branches=i)[[1]]
    tmp <- input[[i]]
    tmp[["experiment"]] <- experiment
    data <- rbind(data, tmp)
  }
  data
}


rjulia_mlm <- function(data, interactions = FALSE){
  # INFO: Julia session stays open once initialized, no current way to restart it, so afex::lmer_alt will replace lme4::lmer as soon as the first case with zerocorr or :: is run.
  julia_library("Parsers, DataFrames, CSV, Plots, MixedModels, RData, CategoricalArrays")
  julia_library("RCall, JellyMe4")
  julia_assign("data", data) # bring data into julia
  if (interactions == FALSE){
    #julia_command("formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + det + base + ar + (1 | subject));")
    julia_command("formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + det + base + ar + (ref + hpf + lpf + emc + mac + det + base + ar | subject));")
  } else {
    julia_command("formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + det + base + ar) ^ 2 + (1 | subject));") ## TODO all variables
  }
  julia_command("model = fit(LinearMixedModel, formula, data);") 
  julia_command("rmodel = (model, data);") # make it a tuple for conversion (Julia model doesn't have the data saved, but R analogue lmer model does); https://github.com/palday/JellyMe4.jl/issues/51, 
  julia_command("RCall.Const.GlobalEnv[:rmodel] = robject(:lmerMod, rmodel);") # alternative to @rput; https://github.com/palday/JellyMe4.jl/issues/72
  rmodel
}


filter_experiment <- function(data){
  experiments = unique(data$experiment)
  data_list = list()
  for (experiment in experiments){
    data_list[[experiment]] <- data[experiment == experiment, ]
  }
  data_list
}

