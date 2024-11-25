
# sandbox: vary one step at a time
library(dplyr)
library(emmeans)
library(ggplot2)

data <- tar_read(data_eegnet)

# mean of accuracy, averaged over participants, but for each other column kept
data <- data %>% 
  # group by everything but subject and accuracy
  group_by(across(-c(subject, accuracy))) %>%
  # calculate mean of accuracy
  
  summarise(across(accuracy, ~ mean(.x, na.rm = TRUE))) %>%
  ungroup()


#mm <- tar_read(eegnet_HLM_emm_means)
# reorder data2, so it looks like the mm data set
# experiment, variable, level, emmean

lookup <- tibble::tibble(
  experiment = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3"),
  emc = c("ica", "ica", "ica", "ica", "ica", "ica", "ica"),
  mac = c("ica", "ica", "ica", "ica", "ica", "ica", "ica"),
  lpf = c("None", "None", "None", "None", "None", "None", "None"),
  hpf = c("0.1", "0.1", "0.1", "0.1", "0.1", "0.1", "0.1"),
  ref = c("P9P10", "P9P10", "P9P10", "average", "P9P10", "P9P10", "P9P10"),
  det = c("None", "None", "None", "None", "None", "None", "None"),
  base = c("200ms", "200ms", "200ms", "200ms", "200ms", "200ms", "200ms"),
  ar = c("int", "int", "int", "int", "int", "int", "int")
)

# Filter data2 based on the lookup table
reference_accuracies <- data %>%
  inner_join(lookup, by = c("experiment", "emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar"))
reference_accuracies

# loop thorugh all combinations of the preprocessing steps
# then get the accuracy for each combination relative to the reference accuracy

results <- data.frame(
  experiment = character(),
  variable = character(),
  level = character(),
  accuracy = numeric()
)

for (e in unique(data$experiment)){
  for (v in c("emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar")){
    for (l in unique(data[[v]])){
      
      # construct the forking path with only this variable changed
      original_lookup <- lookup %>% filter(experiment == e)
      this_forking_path <- original_lookup %>% mutate(!!v := l)
      
      # get the accuracy for the current combination
      this_row <- data %>% filter(experiment == e) %>% inner_join(this_forking_path, by = c("experiment", "emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar"))
      this_acc <- this_row$accuracy
      
      acc_diff <- this_acc - reference_accuracies[reference_accuracies$experiment == e, "accuracy"]
      
      results <- rbind(results, data.frame(experiment = e, variable = v, level = l, accuracy = acc_diff))
      
    }
  }
}

# now re-arrange for heatmap, taken from heatmap function, give it dummy names for now so the function is able to handle it
results2 <- results %>% mutate(emmean = accuracy*100) %>% select(experiment, variable, level, emmean)

heatmap(results2, manual = TRUE)
heatmap(results2, manual = TRUE, unit="\nin absolute\nT-sum")
results2
