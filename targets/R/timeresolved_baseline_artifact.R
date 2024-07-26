
# heatmap across fps (best to worst) of time resolved decoding significances

# DEBUG
#tsums <- tar_read(data_tsum)
#data <- tar_read(data_sliding)

cluster_heatmap <- function(data, tsums){

  # merge tsum onto data
  data <- data %>%
    left_join(tsums, by=c("emc","mac","lpf","hpf","ref","det","base","ar","experiment"))# %>%

  # name each combination of emc mac lpf hpf ref det base ar with a unique number
  data0 <- data %>% 
    mutate(id = paste0(emc,mac,lpf,hpf,ref,det,base,ar)) %>% 
    mutate(id = as.numeric(factor(id))) %>%
    select(-emc,-mac,-lpf,-hpf,-ref,-det,-base,-ar,-p) %>%
    rename(accuracy = `balanced accuracy`) 

  proc_data = data.frame()
  for (thisExperiment in unique(data$experiment)){
    data1 <- data0 %>%
      filter(experiment == thisExperiment) 
    
    # count the number of significance==TRUE values per id, and transfer the rank back to the original dataframe
    data2 <- data1 %>%
      group_by(id) %>%
      summarize(sig_tp = median(tsum), .groups = 'drop') %>% # order according to number of significant time points
      mutate(rank = sig_tp %>% 
               rank(ties.method="first") %>% 
               as.numeric()) %>%
      mutate(rank = max(rank) + 1 - rank) %>% # invert the rank
      left_join(data1, by="id") %>%
      mutate(accuracy = ifelse(significance, accuracy, NaN)) # if significance = false, balanced accuracy = nan
    
    proc_data <- rbind(proc_data, data2)
  }
  
  ## heatmap: y-axis is rank, x-axis is times, fill is significance OR accuracy
  
  # Create a data frame for the vertical lines with legend information
  line_data <- tibble(
    times = c(0),
    event = c("Stimulus or\nResponse Onset")
  )
  max_rank <- max(proc_data$rank, na.rm = TRUE)
  
  proc_data %>% 
    ggplot(aes(x=times, y=rank, fill=accuracy)) + #significance)) +
    geom_tile() +
    scale_fill_viridis_c(option = "magma",
                         na.value = "white") +
    facet_grid(~experiment, scales="free") +
    labs(x="Time [s]", y="Ranked Forking Path") +
    scale_y_reverse(breaks = c(1, 500, 1000, 1500, 2000, max_rank)) +  # Custom y-axis breaks
    scale_x_continuous(breaks = c(-0.5, 0, 0.5)) +  # Custom y-axis breaks
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  +# remove axes grids
    geom_vline(data = line_data, aes(xintercept = times, color = event), 
               linetype = "solid", size = 1) +
    scale_color_manual(values = c("Stimulus or\nResponse Onset" = "green")) # Adjust the color scale for the legend
}

## paired barplot

# dict of preceding intervals
preceding_intervals = list("ERN"= -0.4,
                           "LRP"= -0.4, 
                           "MMN"= 0.0, 
                           "N170"= 0.0, 
                           "N2pc"= 0.0, 
                           "N400"= 0.0, 
                           "P3"= 0.0)


plot_baseline_artifacts <- function(data, tsums){
  # merge tsum onto data
  data <- data %>%
    left_join(tsums, by=c("emc","mac","lpf","hpf","ref","det","base","ar","experiment"))# %>%

  
  data_marked_all = data.frame()
  for (thisExp in unique(data$experiment)){
    
    data_marked <- data %>%
      filter(experiment==thisExp) %>%
      group_by(emc, mac, lpf, hpf, ref, det, base, ar) %>%
      summarize(
        mark = any(times < preceding_intervals[[thisExp]] & significance == TRUE),
        .groups = 'drop'  # This drops the grouping structure after summarizing
      ) 
    
    
    # Reshape data for faceting
    data_long <- data_marked %>%
      pivot_longer(cols = c(emc, mac, lpf, hpf, ref, det, base, ar),
                   names_to = "variable",
                   values_to = "value") %>%
      mutate(variable = factor(variable, levels = c("emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar")))
    
    data_long$experiment = thisExp
    data_marked_all <- rbind(data_marked_all, data_long)
  }
  
  # Create faceted bar plot
  
  data_marked_all %<>%
    # Recode the variable names
    mutate(variable = recode(variable, !!!replacements)) %>%
    relevel_variables(column_name = "value") %>%
    # mark: FALSE to absent, TRUE to present
    mutate(mark = ifelse(mark, "present", "absent"))
  
  p <- ggplot(data_marked_all) +
    geom_bar(aes(x = value, fill = mark), 
             position = position_dodge2(width = 0.4, preserve = "single"), 
             width = 0.7) +
    facet_grid(experiment ~ variable, scales = "free") +
    #facet_wrap(experiment ~ variable, scales = "free_x") +  # Use facet_wrap for free x-axis scales
    #theme_minimal() +
    labs(x = "preprocessing step", y = "Count", fill = "baseline\nartifact") +
    scale_fill_manual(values = c("#4f7871","#851e3e"))
  
  print(p)  
  
}



