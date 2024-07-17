
# heatmap of all model choices and their ranking (top=best model, bottom =worst)
#
#% TODO: best and worst forking paths across experiments in a table in appendix? or a histo-like fashion with some labels?
#  % maybe in a heatmap like fashion from top to bottom (best to worst), left to right (steps), and colors (processing choices). Maybe one could even find patterns in the graph !!
  # if left to right has space, one could put all 7 experiments in facets

seq_pink = c("#e7e1ef", "#c994c7", "#dd1c77")

# own colorscale for factor levels
cols_seq <- c("None" = "black",
          "0.1" = seq_pink[2],    
          "0.5" = seq_pink[3],     
          "6" = seq_pink[3],       
          "20" = seq_pink[2],     
          "45" = seq_pink[1],      
          "ICA" = seq_pink[3],     
          #"200ms" = "black",   
          "200" = seq_pink[2],   
          "400" = seq_pink[3],   
          #"offset" = "black",  
          "linear" = seq_pink[3], 
          #"false" = "black",
          "interp" = seq_pink[2],
          "reject" = seq_pink[3],
          "average" = "black",
          "Cz" = seq_pink[2],
          "P9/P10" = seq_pink[3]
)  


# DEBUG
#library(dplyr)
#library(tidyr)
#library(ggpubr)
#data <- tar_read(data_tsum)

rankampel <- function(data, title=""){
  
  # of forst variable is tsum instead of accuracy, rename for convenience
  if (names(data)[1] == "tsum") {
    data <- data %>% rename(accuracy = tsum)
  }
  
  # average off accuracy across subs
  result <- data %>%
    group_by(emc, mac, lpf, hpf, ref, det, base, ar, experiment) %>%
    summarize(avg_accuracy = mean(accuracy), .groups = 'drop')
  
  # order the data by avg_accuracy descending (per experiment)
  result <- result %>%
    group_by(experiment) %>%
    arrange(desc(avg_accuracy)) %>%
    mutate(rank = row_number()) #%>%
    #ungroup()
  
  
  #result_ERN <- result %>% filter(experiment == "ERN") %>% select(-c(avg_accuracy, rank, experiment))
  result <- result %>% select(-c(avg_accuracy, rank)) #, rank
  
  # Transform the data into a long format
  data_long <- result %>%
    group_by(experiment) %>%
    mutate(Row = row_number()) %>% #row_number()
    pivot_longer(cols = -c(experiment,Row), names_to = "Column", values_to = "Value") %>%
    ungroup()
    #mutate(experiment=result$experiment)
    #head(200)
  
  # order the steps according to their temporal MV order
  data_long$Column = factor(data_long$Column, level=c("emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar"))
  
  
  data_long_rec <- data_long %>% 
    mutate(Column = recode(Column, !!!replacements_sparse)) %>%
    mutate(Value = recode(Value, !!!replacements_sparse))
  
  
  # Create the tile plot
  p1 <- ggplot(data_long_rec, aes(x = Column, y = Row, fill = Value)) + # y = as.factor(Row)
    geom_tile() + # color = "white"
    scale_y_reverse() + # reverse y, so that best pipeline is top, worst is bottom
    scale_fill_manual(values = cols_seq) + #c("Low" = "blue", "Medium" = "yellow", "High" = "red")) +
    theme_minimal() +
    labs(title = title,#"Ordered Forking Paths", 
         x = "Processing Step", 
         y = "Ranked Forking Path", 
         fill = "Processing\nChoice") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    theme(legend.position="none") +
    facet_grid(.~experiment) #, scales = "free_x"
  #p1
  
  # get legends for each processing step
  leg_data <- data_long_rec %>%
    filter(experiment=="ERN")
  unique_steps <- unique(leg_data$Column)
  legends = list()
  for (step in unique_steps){
    
    this_leg_data <- leg_data %>% filter(Column == step)
    
    # relevel LPF order, TODO: also do it with others for consistency, event tho no effect
    if (step=="low pass"){
      # reorder the factor levels of the variables in the following order
      new_order = c("6", "20", "45","None") # TODO double check if it is correct with the new MV3
      this_leg_data[[Column]] <- factor(this_leg_data[[Column]], levels = new_order)  
    }
    
    ptmp <- ggplot(this_leg_data, aes(x = Column, y = Row, fill = Value)) + 
      geom_tile() + 
      scale_fill_manual(values = cols_seq) + 
      theme_minimal() +
      labs(fill=step) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    legend <- ggpubr::as_ggplot(ggpubr::get_legend(ptmp))
    legends <- c(legends, list(legend))
  }
  #legends
  
  # put legends on plot
  cow <- cowplot::ggdraw() + 
    cowplot::draw_plot(p1, x = 0, y = 0.25, width = 1.0, height = 0.75) 
  
  d <- 1/8
  for (i in 0:7){
    # single legends
    cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                    x = d*i+0.05, #+0.05, 
                                    y = 0.1, ##0.9-d*i, 
                                    width = 0.03, height = 0.03)
  }
  cow
}



# DEBUG
# data1 = tar_read(data_eegnet)
# data2 = tar_read(data_tsum)
# title1="EEGNet"
# title2="Time-resolved"

rankampel_merge <- function(data1, data2, title1="", title2=""){
  # rename tsum to accuracy
  data2 <- data2 %>% rename(accuracy = tsum)
  legends = list()
  plots = list()
  data <- list(data1, data2)
  title <- list(title1, title2)
  
  # DEBUG
  # it=1
  for (it in c(1,2)){
    thisData = data[[it]]
    thisTitle = title[[it]]
    # average off accuracy across subs
    result <- thisData %>%
      group_by(emc, mac, lpf, hpf, ref, det, base, ar, experiment) %>%
      summarize(avg_accuracy = mean(accuracy), .groups = 'drop')
    
    # order the data by avg_accuracy descending (per experiment)
    result <- result %>%
      group_by(experiment) %>%
      arrange(desc(avg_accuracy)) %>%
      mutate(rank = row_number()) #%>%
    #ungroup()
    
    
    #result_ERN <- result %>% filter(experiment == "ERN") %>% select(-c(avg_accuracy, rank, experiment))
    result <- result %>% select(-c(avg_accuracy, rank)) #, rank
    
    # Transform the data into a long format
    data_long <- result %>%
      group_by(experiment) %>%
      mutate(Row = row_number()) %>% #row_number()
      pivot_longer(cols = -c(experiment,Row), names_to = "Column", values_to = "Value") %>%
      ungroup()
    #mutate(experiment=result$experiment)
    #head(200)
    
    # order the steps according to their temporal MV order
    data_long$Column = factor(data_long$Column, level=c("emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar"))
    
    
    data_long_rec <- data_long %>% 
      mutate(Column = recode(Column, !!!replacements_sparse)) %>%
      mutate(Value = recode(Value, !!!replacements_sparse))
    
    
    # Create the tile plot
    p1 <- ggplot(data_long_rec, aes(x = Column, y = Row, fill = Value)) + # y = as.factor(Row)
      geom_tile() + # color = "white"
      scale_y_reverse() + # reverse y, so that best pipeline is top, worst is bottom
      scale_fill_manual(values = cols_seq) + #c("Low" = "blue", "Medium" = "yellow", "High" = "red")) +
      theme_minimal() +
      labs(title = thisTitle,#"Ordered Forking Paths", 
           x = "Processing Step", 
           y = "Ranked Forking Path", 
           fill = "Processing\nChoice") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      theme(legend.position="none") +
      facet_grid(.~experiment) #, scales = "free_x"
    
    plots = c(plots, list(p1))
    #p1
    
    if (length(legends) == 0){
      # get legends for each processing step
      leg_data <- data_long_rec %>%
        filter(experiment=="ERN")
      unique_steps <- unique(leg_data$Column)
      legends = list()
      
      legend_size= 0.2 # cm
      legend_title_size = 10
      legend_text_size = 8
      
      for (step in unique_steps){
        this_leg_data <- leg_data %>% filter(Column == step)
        
        #print(this_leg_data)
        # relevel LPF order, TODO: also do it with others for consistency, event tho no effect
        if (step=="low pass"){
          # reorder the factor levels of the variables in the following order
          new_order = c("6", "20", "45","None") # TODO double check if it is correct with the new MV3
          this_leg_data$Value <- factor(this_leg_data$Value, levels = new_order)  
        }
        
        ptmp <- ggplot(this_leg_data, aes(x = Column, y = Row, fill = Value)) + 
          geom_tile() + 
          scale_fill_manual(values = cols_seq) + 
          theme_minimal() +
          labs(fill=step) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1),
                legend.key.size = unit(legend_size, "cm"),  # Adjust legend key size
                legend.title = element_text(size = legend_title_size),
                legend.text = element_text(size = legend_text_size))  # Adjust legend text size
        legend <- ggpubr::as_ggplot(ggpubr::get_legend(ptmp))
        legends <- c(legends, list(legend))
      }
      #legends
    }
  }
  
  # put legends on plot
  cow <- cowplot::ggdraw() + 
    cowplot::draw_plot(plots[[1]], x = 0, y = 0.56, width = 1.0, height = 0.44) +
    cowplot::draw_plot(plots[[2]], x = 0, y = 0.12, width = 1.0, height = 0.44)
  
  d <- 1/8
  for (i in 0:7){
    # single legends
    cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                    x = d*i+0.05, #+0.05, 
                                    y = 0.04, ##0.9-d*i, 
                                    width = 0.01, # TODO: these parametes dont change anything, if legend should be smaller, then change this during creating the single ones
                                    height = 0.01)
  }
  cow
}