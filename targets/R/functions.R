get_preprocess_data <- function(file) {
  data <- read_csv(file, col_types = cols())
  # new, preprocess already
  data$hpf <- factor(data$hpf, levels = c("None", "0.1", "0.5"))
  data$lpf <- factor(data$lpf, levels = c("None", "6", "20", "45"))
  data$ref <- factor(data$ref, levels = c("average", "Cz", "P9P10"))
  data$emc <- factor(data$emc, levels = c("None", "ica"))
  data$mac <- factor(data$mac, levels = c("None", "ica"))
  data$base <- factor(data$base, levels = c("200ms", "400ms"))
  data$det <- factor(data$det, levels = c("offset", "linear"))
  data$ar <- factor(data$ar, levels = c("FALSE", "TRUE"))
  data$experiment <- factor(data$experiment, levels = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3", "LRP_6-9", "LRP_10-11", "LRP_12-13", "LRP_14-17", "LRP_18+"))
  data$dataset <- factor(data$dataset)
  data
}

# factorize <- function(data) {
#   data$hpf <- factor(data$hpf, levels = c("None", "0.1", "0.5"))
#   data$lpf <- factor(data$lpf, levels = c("None", "6", "20", "45"))
#   data$ref <- factor(data$ref, levels = c("average", "Cz", "P9P10"))
#   data$emc <- factor(data$emc, levels = c("None", "ica"))
#   data$mac <- factor(data$mac, levels = c("None", "ica"))
#   data$base <- factor(data$base, levels = c("200ms", "400ms"))
#   data$det <- factor(data$det, levels = c("offset", "linear"))
#   data$ar <- factor(data$ar, levels = c("FALSE", "TRUE"))
#   data$experiment <- factor(data$experiment)
#   data
# }

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

paired_tests <- function(data, study="ERPCORE"){
  # if we are in the MIPDB data, then for the experiment variable can not be grouped for tests,
  # therefore, delete the experiment variable, and make an unpaired test for it
  # 1. only try by removing experiment in MIPDB
  
  if (study=="MIPDB"){
    data_exp <- data %>% filter(variable == "experiment") # NEW
    data <- data %>% filter(variable != "experiment")
  }
  
  results <- data %>%
    group_by(variable) %>%
    pairwise_t_test(accuracy ~ factor, paired = TRUE, p.adjust.method = "BY",
                    pool.sd = FALSE, detailed = TRUE)

  if (study=="MIPDB"){
    results_exp <- data_exp %>%
      group_by(variable) %>%
      pairwise_t_test(accuracy ~ factor, paired = FALSE, p.adjust.method = "BY",
                      pool.sd = FALSE, detailed = TRUE) %>%
      select(-estimate1, -estimate2)
    results <- rbind(results, results_exp)
  }
  
  results
}

# rename variables
replacements <- list(
  "hpf" = "high pass filter (Hz)",
  "lpf" = "low pass filter (Hz)",
  "ref" = "reference electrode",
  "ar" = "autoreject",
  "mac" = "muscle artifact correction",
  "emc" = "eye movement correction",
  "base" = "baseline correction",
  "det" = "detrending",
  "0.1" = "0.1Hz",
  "0.5" = "0.1Hz",
  "6" = "6Hz",
  "20" = "20Hz",
  "45" = "45Hz",
  "FALSE" = "False",
  "TRUE" = "True",
  "ica" = "ICA",
  "P9P10" = "P9 / P10"
)

raincloud_mm <- function(data, title = ""){
  # Apply replacements batchwise across all columns
  data <- data %>%
    mutate(variable = recode(variable, !!!replacements))
  
  # https://rpubs.com/rana2hin/raincloud
  ggplot(data, aes(x = factor, y = accuracy)) +
    
    # add half-violin from {ggdist} package
    stat_halfeye(
      # adjust bandwidth
      adjust = 0.5,
      # move to the right
      justification = -0.2,
      # remove the slub interval
      .width = 0,
      point_colour = NA,
      scale = 0.5 ##  <(**new parameter**)
    ) +
    geom_boxplot(
      width = 0.12,
      # removing outliers
      outlier.color = NA,
      alpha = 0.5
    ) +
    stat_dots(
      # ploting on left side
      side = "left",
      # adjusting position
      justification = 1.1,
      # adjust grouping (binning) of observations
      binwidth = 0.005,
      
    ) +
    facet_wrap(~variable, scales="free") +
    labs(title = title,
         x="preprocessing step",
         y="accuracy")

}

paired_box <- function(data, title=""){
  # Apply replacements batchwise across all columns
  data <- data %>%
    mutate(variable = recode(variable, !!!replacements))
  
  # boxplot with paired point plot
  ggpaired(filter(data, variable != "experiment"), # exclude the experiment variable if present
           x = "factor", 
           y = "accuracy",
           id = "subject",
           line.color = "subject", #"gray",
           title = title,
           line.size = 0.6
           #facet.by = c("variable") # this one does not work for different factors per facet
  ) + 
    facet_grid(.~variable, scales="free") +
    theme(legend.position = "none") +
    labs(x="preprocessing step",
         y="accuracy")
}


filter_experiment <- function(data){
  experiments = unique(data$experiment)
  data_list = list()
  for (experiment in experiments){
    data_list[[experiment]] <- data[experiment == experiment, ]
  }
  data_list
}



## HLM and EMM


#hlm <- function(data, formula) {
#  lmer(formula, data = data)
#}

# est_emm <- function(model, variables){
#   for (variable in variables){
#     # MAIN EFFECTS (1 factor)
#     emm <- emmeans(hlm, specs = formula(paste0(c("pairwise ~ ",variable)))) # ref, within exp
#     emm$emmeans %>% as.data.frame() # leaving out contrasts for now
#   }
# }
# 
# plot_emm <- function(model, variables){
#   
#   
# }
# 
# 
# find_significant_combinations <- function(features, contrast_df) {
#   # Initialize an empty list to store significant combinations
#   significant_combinations <- list()
#   
#   # Loop over unique experiment combinations
#   for (i in 1:(length(features) - 1)) {
#     for (j in (i + 1):length(features)) {
#       feature1 <- features[i]
#       feature2 <- features[j]
#       
#       # Check if both experiments are present in con$contrast
#       if (any(grepl(feature1, contrast_df$contrast)) && any(grepl(feature2, contrast_df$contrast))) {
#         # Get the row indices where the experiments are found in con$contrast
#         indices1 <- grep(feature1, contrast_df$contrast)
#         indices2 <- grep(feature2, contrast_df$contrast)
#         
#         # Check if there are any significant pairs
#         significant_pairs <- intersect(indices1, indices2)
#         if (length(significant_pairs) > 0) {
#           # Get the corresponding p-values
#           p_values <- contrast_df$p.value[significant_pairs]
#           
#           # Check if any p-value is smaller than 0.05
#           if (any(p_values < 0.05)) {
#             # Add the significant combination to the list
#             significant_combinations[[length(significant_combinations) + 1]] <- c(as.character(feature1), as.character(feature2))
#           }
#         }
#       }
#     }
#   }
#   return(significant_combinations)
# }
