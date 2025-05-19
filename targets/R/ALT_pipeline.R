
# functions to use with the alternative pipeline order
# 

get_preprocess_data_ALT <- function(file) {
  data <- read_csv(file, col_types = cols())
  
  # change column order for arbitrary reason: None in LPF should be last, but because None is a factor both in hpf in lpf, lpf should come first with none as last entry, then later in the analyses when putting both in a long dataframe it will be ordered correctly
  #data <- data %>%
  #  select(names(data)[1:2], lpf, hpf, everything()) 
  # but keep None first here, because this will be the reference for HLM, and makes it therefore easier to interpret
  # recode lpf levels later
  # DOESNT work becuase the variables are ordered by name later and HPF will be BEFORE LPF...
  
  
  # new, preprocess already
  data$hpf <- factor(data$hpf, levels = c("None", "0.1", "0.5"))
  data$lpf <- factor(data$lpf, levels = c("None", "6", "20", "45"))
  data$ref <- factor(data$ref, levels = c("average", "Cz", "P9P10"))
  data$emc <- factor(data$emc, levels = c("None", "ica"))
  data$mac <- factor(data$mac, levels = c("None", "ica"))
  data$base <- factor(data$base, levels = c("200ms", "400ms")) #"None", 
  data$det <- factor(data$det, levels = c("offset", "linear"))
  #data$ar <- factor(tolower(data$ar), levels = c("false", "true"))
  data$ar <- factor(data$ar, levels = c("FALSE", "TRUE"))
  data$experiment <- factor(data$experiment, levels = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")) #, "LRP_6-9", "LRP_10-11", "LRP_12-13", "LRP_14-17", "LRP_18+", "6-9", "10-11", "12-13", "14-17", "18+"
  
  data$dataset <- factor(data$dataset)
  # discard MIPDB dataset
  data <- subset(data, dataset != "MIPDB")
  
  if ("dataset" %in% names(data)) {
    data <- subset(data, select = -c(dataset))
  }
  
  # new: replace with paper-ready variable names / factor levels
  # col names
  #names(data) <- recode(names(data), !!!replacements)
  # NOT DONE, as this would disrupt short variable naming during modeling
  
  data
}

reorder_variables_ALT <- function(data, column_name){
  # reorder the factor levels of the variables in the following order
  #new_order = c("ref", "hpf","lpf","emc","mac","det","base","ar") # original
  new_order = c("ref","lpf","hpf","emc","mac","det","base","ar") #c("ref", "lpf","hpf","emc","mac","det","base","ar") # I CHANGED HPF AND LPF
  data[[column_name]] <- factor(data[[column_name]], levels = new_order)  
  return(data)
}

relevel_variables_ALT <- function(data, column_name){
  # reorder the factor levels of the variables in the following order
  new_order = c("average", "Cz", "P9P10", "6", "20", "45","None","0.1", "0.5","ica", "200ms", "400ms", "offset", "linear", "FALSE", "TRUE") 
  data[[column_name]] <- factor(data[[column_name]], levels = new_order)  
  return(data)
}


# rename variables
replacements_ALT <- list(
  "hpf" = "high-pass", # [Hz]
  "lpf" = "low-pass", # [Hz]
  "ref" = "reference",
  "ar" = "autoreject",
  "mac" = "muscle",
  "emc" = "ocular",
  "base" = "baseline",
  "det" = "detrending",
  "0.1" = "0.1 Hz",
  "0.5" = "0.5 Hz",
  "6" = "6 Hz",
  "20" = "20 Hz",
  "45" = "45 Hz",
  #"FALSE" = "False",
  #"false" = "False",
  "FALSE" = "None",
  "false" = "None",
  "TRUE" = "True",
  "true" = "True",
  "int" = "interpolate",
  "intrej" = "reject",
  "ica" = "ICA",
  "200ms" = "200 ms",
  "400ms" = "400 ms",
  "ica" = "ICA",
  "P9P10" = "P9/P10"
)
replacements_sparse_ALT <- list(
  "hpf" = "high-pass", # [Hz]
  "lpf" = "low-pass", # [Hz]
  "ref" = "reference",
  "ar" = "autoreject",
  "mac" = "muscle",
  "emc" = "ocular",
  "base" = "baseline",
  "det" = "detrending",
  #"0.1" = "0.1 Hz",
  #"0.5" = "0.5 Hz",
  #"6" = "6 Hz",
  #"20" = "20 Hz",
  #"45" = "45 Hz",
  "FALSE" = "None",
  "false" = "None",
  "TRUE" = "interp",
  "true" = "interp",
  "int" = "interp",
  "intrej" = "reject",
  "ica" = "ICA",
  "200ms" = "200",
  "400ms" = "400",
  "ica" = "ICA",
  "P9P10" = "P9/P10"
)


heatmap_ALT <- function(data){
  data <- data %>% 
    reorder_variables_ALT(column_name = "variable") %>%
    relevel_variables_ALT(column_name = "level") %>% 
    # Apply replacements batchwise across all columns
    mutate(variable = recode(variable, !!!replacements_ALT)) %>%
    # NEW: replacements for some levels, to not overload the image too much
    mutate(level = recode(level, !!!replacements_sparse_ALT)) %>%
    # delete the experiment compairson in the full data
    #filter(!(experiment == "ALL" & variable == "experiment")) %>% 
    # center around zero for better comparability
    group_by(experiment) %>%
    mutate(emmean = (emmean / mean(emmean) - 1) * 100 ) # now it is percent
  
  ggplot(data, aes(y = 0, x = level, fill = emmean)) +
    geom_tile() +
    #geom_text(aes(label = sprintf("%.1f", emmean)), size = 3) + # Add text labels with one decimal place
    geom_text(aes(label = sprintf("%+.1f", emmean)), size = 3) + # Add text labels with one decimal place and + sign for positives
    facet_grid(experiment~variable, scales="free") +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          strip.text.y = element_text(angle=0)) + # rotate the experiment labels
  scale_fill_gradient2(low=colors_dark[1], mid="white", high=colors_dark[2]) + 
    labs(x="Preprocessing step",
         y="",
         #fill="% change\naccuracy")  
         fill="% Deviation from\nmarginal mean")  

}

