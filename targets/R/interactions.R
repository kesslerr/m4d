# Function to replace lower diagonal values with NA, needed for interaction_plot
upper_diagonal_only <- function(df) {
  df %>%
    mutate(emmean = ifelse(as.numeric(variable.2) > as.numeric(variable.1), NA, emmean))
}

interaction_plot <- function(means, title_prefix=""){
  # DEBUG
  #means = tar_read(eegnet_HLMi2_emm_means, branches=1)
  #means <- means %>% filter(experiment=="ERN")
  
  experiment <- unique(means$experiment)
  means %<>% 
    reorder_variables("variable.1") %>%
    reorder_variables("variable.2") %>%
    relevel_variables("level.1") %>%
    relevel_variables("level.2") 
  #means <- 
  meansr <- means %>% 
    mutate(variable.1 = recode(variable.1, !!!replacements_sparse)) %>%
    mutate(variable.2 = recode(variable.2, !!!replacements_sparse)) %>%
    mutate(level.1 = recode(level.1, !!!replacements_sparse)) %>%
    mutate(level.2 = recode(level.2, !!!replacements_sparse))  
  

  only_upper_diag=TRUE
  
  if (only_upper_diag==TRUE){
    # filter diagonal
    meansr_filtered <- upper_diagonal_only(meansr)
    
    # filter the lowest facet (AR), if diagonal is removed during plotting (no new data in there)
    meansr_filtered %<>% 
      filter(variable.2 != "autoreject")
  }  else {
    meansr_filtered <- meansr
  }
  # only upper diag plot
  p1 <- ggplot(meansr_filtered, 
               aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
    geom_line(linewidth = 1.2) + 
    facet_grid(variable.2 ~ variable.1, scales = "free") +
    labs(title = paste0(title_prefix, experiment),
         y = "Marginal Mean", 
         x = "Preprocessing Step", 
         #color = "Group: ") + 
         color = "") + 
    scale_color_manual(values = cols) +
    theme_classic() +
    scale_x_discrete(expand = c(0.2, 0.0)) + 
    theme(legend.position = "none")  # Remove legend
  
  # make pseudo plots for each row, and extract only the legend
  variable.2s <- sort(unique(meansr$variable.2))
  legends <- list()
  for (v2 in variable.2s){
    results_filtered <- meansr %>% filter(variable.2 == v2)
    ptmp <- ggplot(results_filtered, 
                   aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
      geom_line(size = 1.2) + 
      facet_grid(.~variable.1, scales = "free") +
      #labs(color = paste0("Group: ",v2)) +
      labs(color = paste0("",v2)) +
      scale_color_manual(values=cols) +
      theme_classic()
    # get legend
    legend <- as_ggplot(ggpubr::get_legend(ptmp))
    legends <- c(legends, list(legend))
  }
  
  # TODO, maybe significances in the lower diag, or in the plot itself
  
  ## legends on diagonals
  cow <- cowplot::ggdraw() + 
    cowplot::draw_plot(p1, x = 0, y = 0, width = 1.0, height = 1.0) 
  
  if (only_upper_diag==FALSE){
    d <- 1/8.5
    e <- 1/9
    for (i in 0:7){
      # single legends
      cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                      x = d*i+0.05, 
                                      y = 0.9-d*i, 
                                      width = 0.1, height = 0.03)
      # horizontal lines between facets
      if (i<7){
        cow <- cow + cowplot::draw_line(x = c(0.05, 0.97), 
                                        y = c(0.85-d*i, 0.85-d*i), 
                                        color = "grey", size = 0.5)
      }
    }
  } else {
    c <- 1/8.5 # x 
    d1 <- 1/7.6 # y leg
    d2 <- 1/7.5 # y line
    e <- 1/8
    for (i in 0:6){
      # single legends
      cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                      x = c*i+0.05, 
                                      y = 0.88-d1*i, 
                                      width = 0.1, height = 0.03)
      # horizontal lines between facets
      if (i<6){
        cow <- cow + cowplot::draw_line(x = c(0.05, 0.97), 
                                        y = c(0.83-d2*i, 0.83-d2*i), 
                                        color = "grey", size = 0.5)
      }
    }
  }

  cow
}
