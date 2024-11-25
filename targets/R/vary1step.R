
varyOneCalculation <- function(data){
  # mean of accuracy, averaged over participants, but for each other column kept
  
  # if accuracy in col names
  if ("accuracy" %in% colnames(data)){
    data <- data %>% 
      group_by(across(-c(subject, accuracy))) %>%
      summarise(across(accuracy, ~ mean(.x, na.rm = TRUE))) %>%
      ungroup()
  }
  # for tsum it's already a group result!
  
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
  #reference_accuracies
  
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
        
        if (identical(this_forking_path, original_lookup)==TRUE){
          results <- rbind(results, data.frame(experiment = e, variable = v, level = l, accuracy = NA))
          next
        }
        
        # get the accuracy for the current combination
        this_row <- data %>% filter(experiment == e) %>% inner_join(this_forking_path, by = c("experiment", "emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar"))
        
        if ("accuracy" %in% colnames(data)){
          this_acc <- this_row$accuracy
          acc_diff <- this_acc - reference_accuracies[reference_accuracies$experiment == e, "accuracy"]
        } else if ("tsum" %in% colnames(data)){
          this_acc <- this_row$tsum
          acc_diff <- this_acc - reference_accuracies[reference_accuracies$experiment == e, "tsum"]
          acc_diff <- acc_diff[[1]]
        }
        
        results <- rbind(results, data.frame(experiment = e, variable = v, level = l, accuracy = acc_diff))
        
      }
    }
  }
  
  results
  
} 