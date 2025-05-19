# Function to replace lower diagonal values with NA, needed for interaction_plot
upper_diagonal_only <- function(df) {
  df %>%
    mutate(emmean = ifelse(as.numeric(variable.2) > as.numeric(variable.1), NA, emmean))
}

interaction_plot <- function(means, title_prefix="", omni_data){
  # DEBUG
  #means = tar_read(sliding_LMi2_emm_means, branches=3)
  #means <- means %>% filter(experiment=="MMN")
  # omni_data = tar_read(eegnet_HLM_emm_omni)
  
  this_experiment <- unique(means$experiment)
  
  # R1: label the y-axes more specifically
  if (grepl("EEGNet", title_prefix)) {
    y_label <- "Accuracy"
  } else if(grepl("Time-resolved", title_prefix)){
    y_label <- "T-sum"
  } else {
    # throw error
    stop("Unknown title prefix, can not extrac y-label from it.")
  }
  
  means %<>% 
    reorder_variables("variable.1") %>%
    reorder_variables("variable.2") %>%
    relevel_variables("level.1") %>%
    relevel_variables("level.2") 
  
  meansr <- means %>% 
    mutate(variable.1 = recode(variable.1, !!!replacements_sparse)) %>%
    mutate(variable.2 = recode(variable.2, !!!replacements_sparse)) %>%
    mutate(level.1 = recode(level.1, !!!replacements_sparse)) %>%
    mutate(level.2 = recode(level.2, !!!replacements_sparse))  
  
  # R1: get the significances
  omni_data <- omni_data %>%
    filter(experiment == this_experiment) %>%
    # model_term: only use the rows that contain ":"
    filter(grepl(":", `model term`)) %>%
    # split model term based on ":"
    mutate(
      term = str_split(`model term`, ":"),
      term1 = map_chr(term, 1),
      term2 = map_chr(term, 2)
    ) %>%
    # rename term1 and term2 to real names
    mutate(
      term1 = recode(term1, !!!replacements_sparse),
      term2 = recode(term2, !!!replacements_sparse)
    ) %>%
    # remove . in sign.unc and sign.fdr
    mutate(
      sign.unc = str_remove(sign.unc, "\\."),
      sign.fdr = str_remove(sign.fdr, "\\.")
    )

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
  
  # R1: merge omni_data on means_r_riltered based on variable.1 and variable.2 and term1 and term2
  meansr_filtered <- meansr_filtered %>%
    left_join(omni_data, by = c("variable.1" = "term2", "variable.2" = "term1"), 
              keep = TRUE) # keeps the ordering of the factors

  # R1: for centering the asterisks based on the x values of a facet
  # Calculate the number of unique x-values per facet
  facet_x_centers <- meansr_filtered %>%
    group_by(variable.1, variable.2) %>%
    summarise(n_x = n_distinct(level.1), .groups = "drop") %>%
    mutate(center_x = (n_x+1) / 2)  # Find the center (half of the number of levels)
  meansr_filtered <- meansr_filtered %>%
    left_join(facet_x_centers, by = c("variable.1", "variable.2"))
  
  # only upper diag plot
  p1 <- ggplot(meansr_filtered, 
               aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
    geom_line(linewidth = 1.2) + 
    #geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.2, linewidth = 0.8) + # R1: add errorbars
    #geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.2, linewidth = 0.8) + # R1: add errorbars
    # TODO different CL code for eegnet and sliding 
    # TODO: don't use CIs: https://cran.r-project.org/web/packages/emmeans/vignettes/comparisons.html
    facet_grid(variable.2 ~ variable.1, scales = "free_x") +
    # R1: plot significance asterisks in facets
    geom_text(aes(x = center_x, 
                  y = max(emmean, na.rm = TRUE) - (max(emmean, na.rm = TRUE) - min(emmean, na.rm = TRUE)) * 0.1, 
                  label = sign.unc), 
              inherit.aes = FALSE, size = 5) +
    labs(title = paste0(title_prefix, this_experiment),
         y = y_label, #"Marginal mean", 
         x = "Preprocessing step", 
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
      theme_classic() +
      # new, make legend transparent to not overlay lines later in cowplot?
      theme(legend.background = element_rect(fill = NA),  # Transparent background
            legend.key = element_rect(fill = NA))         # Transparent key boxes
    # get legend
    legend <- as_ggplot(ggpubr::get_legend(ptmp))
    legends <- c(legends, list(legend))
  }
  
  ## legends on diagonals
  cow <- cowplot::ggdraw() + 
    cowplot::draw_plot(p1, x = 0, y = 0, width = 1.0, height = 1.0) 
  
  if (only_upper_diag==FALSE){
    d <- 1/8.5
    e <- 1/9
    for (i in 0:7){
      # horizontal lines between facets
      if (i<7){
        cow <- cow + cowplot::draw_line(x = c(0.07, 0.95),  # horizontal extend
                                        y = c(0.85-d*i, 0.85-d*i), # 
                                        color = "grey", size = 0.5)
      }
      # single legends: new: draw after line, so line not gets hidden
      cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                      x = d*i+0.05, 
                                      y = 0.9-d*i, 
                                      width = 0.1, height = 0.03)
    }
  } else {
    c <- 1/8.5 # x 
    d1 <- 1/7.8 # y leg
    d2 <- 1/7.75 # y line, davor 7.5
    e <- 1/8
    for (i in 0:6){
      # horizontal lines between facets
      if (i<6){
        cow <- cow + cowplot::draw_line(x = c(0.07, 0.95),  # horizontal extend
                                        y = c(0.82-d2*i, 0.82-d2*i), 
                                        color = "grey", size = 0.5)
      }
      # single legends: new: draw after line, so line not gets hidden
      cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                      x = c*i+0.05, 
                                      y = 0.865-d1*i, 
                                      width = 0.1, height = 0.03)
    }
  }
  cow
}
