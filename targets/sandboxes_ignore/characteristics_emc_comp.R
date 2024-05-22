
# find bad muscles analysis

#ICA = c("EOG", "EMG")

# debug
ICA = "EOG"

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
ggplot(results, aes(x=lpf, y=components)) + 
  geom_boxplot() +
  labs(y="# of dropped components", x="Low-pass filter (Hz)", title = "Muscle artifact correction via ICA") +
  facet_grid(experiment ~ .)

# ICA
# ggplot(results, aes(x=ref, y=components)) + 
#   geom_boxplot() +
#   labs(y="# of dropped components", x="Low-pass filter (Hz)", title = "Muscle artifact correction via ICA") +
#   facet_grid(experiment ~ .)