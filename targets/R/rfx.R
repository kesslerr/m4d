
# RFX vis
rfx_vis <- function(model, orig_data){
  data <- ranef(model)$subject
  
  data_long <- data %>%
    pivot_longer(
      cols = names(.), #-c("subject"), #, # Select columns starting with "est"
      names_to = "level",         # Create the "level" column
      values_to = "mean" # Create the "conditional mean" column
    ) 
  if (any(startsWith(data_long$level, "experiment"))) {
    title <- "ALL" 
  } else {
    title <- unique(orig_data$experiment)
  } 
  ggplot(data_long,
         aes(y=mean, x=level)) +
    geom_boxplot() +
    labs(y="Conditional mean", x="Random effects term", title=title) +
    theme(axis.text.x = element_text(angle=90))
  
}

# RFX and sociodemographics
plot_rfx_demographics <- function(model, demographics, orig_data, framework){
  # DEBUG
  #model <- tar_read(eegnet_HLMi2, branches=1)[[1]]
  #demographics <- tar_read(demographics)
  #orig_data <- tar_read(data_eegnet_exp) %>% filter(experiment == "ERN")
  
  # capitalize first letter of factor variable: handedness
  demographics$handedness <- factor(demographics$handedness,
                                    levels = c("left", "right"),
                                    labels = c("Left", "Right"))
  
  # from rfx_vis function
  data <- lme4::ranef(model)$subject %>%
    mutate(Subject = rownames(.)) %>%
    mutate(Intercept = `(Intercept)`) %>%
    select(c(Intercept, Subject))
  rownames(data) <- NULL
  experiment <- unique(orig_data$experiment)
  
  # merge with demographics
  data <- left_join(data, demographics, c("Subject" = "participant_id"))
  
  # plot age
  p1 <- ggplot(data, aes(x=age, y=Intercept, color=sex)) +
    geom_point() +
    #geom_smooth(method="lm", se=TRUE) +
    geom_hline(aes(yintercept=0), lty="dashed") +
    labs(x="Age", y="Intercept") +
    scale_color_manual(values = colors_dark)
  
  p2 <- ggplot(data, aes(x=sex, y=Intercept, fill=sex)) +
    geom_boxplot(notch=FALSE) +
    geom_hline(aes(yintercept=0), lty="dashed") +
    labs(x="Sex", y="Intercept") +
    guides(fill = "none") +# remove legend for "fill"
    scale_fill_manual(values = colors_dark)
  
  p3 <- ggplot(data, 
               aes(x=Intercept, fill=handedness)) +
    geom_histogram(bins=20) + 
    geom_vline(aes(xintercept=0), lty="dashed") +
    labs(x="Intercept", y="Participant count", fill="Handedness") +
    #scale_fill_viridis_d(begin=0, end=0.8) +
    scale_fill_manual(values = c(colors_light[3], "grey"))
  
  #If it is not the first (ERN) plot, then remove all legends
  if (experiment == "ERN"){
    
    
    # workaround, to make the squashed xticklabels better visible
    if (framework == "Time-resolved"){
      p3 <- p3 + scale_x_continuous(breaks = c(-0.1, 0, 0.1))  # Set custom x-ticks
    }
    
    #p1 <- p1 + theme(legend.position = c(0.1, 0)) # position within figure bottom left
    #p3 <- p3 + theme(legend.position = c(0.1, 1)) # postition within figure top left
  } else {
    p3 <- p3 + theme(legend.position="none")
  } 
  p1 <- p1 + theme(legend.position="none")
  
  # If it is not the last (P3) plot, remove all X labels
  if (experiment != "P3"){
    p1 <- p1 + labs(x="")
    p2 <- p2 + labs(x="")
    p3 <- p3 + labs(x="")
  }
  
  # statistics on the results
  # 1. 2-sample test on male vs female with BH correction (7 experiments)
  males <- data[data$sex == "Male", ]
  females <- data[data$sex == "Female", ]
  
  # t test
  #t_result <- t.test(males$Intercept, females$Intercept, var.equal = TRUE)
  #adj_p <- p.adjust(t_result$p.value, method="BH", n=7) # alternative: bonferroni
  
  # wilcox test / man whitney u test; groups are small
  w_result = wilcox.test(males$Intercept, females$Intercept, 
              alternative = "two.sided",
              paired = FALSE,
              exact = FALSE, 
              correct = FALSE, 
              conf.int = FALSE)
  adj_p <- p.adjust(w_result$p.value, method="BH", n=7) # alternative: bonferroni
  
  if (adj_p < 0.995){
    p2 <- p2 + annotate("text", x=1.5, y=0.15, label=paste("p=", sprintf("%.2f", adj_p)), color="black", size=4)
  } else {
    p2 <- p2 + annotate("text", x=1.5, y=0.15, label=paste("p≈", sprintf("%.2f", adj_p)), color="black", size=4)
  }
  # 2. are lines in p1 stat diff
  #agemodel <- lm(Intercept ~ age * sex, data = data)
  #if used, then interpret the interaction term but maybe also the slope
  
  ggarrange(p1,p2,p3, ncol=3) %>%
    annotate_figure(left = text_grob(experiment, 
                                     color = "black", face = "bold", size = 12, rot=90))
}

# extract ranef from all experiment HLM models
extract_rfx_exp <- function(model, orig_data){
  data <- ranef(model)$subject %>%
    mutate(Subject = rownames(.)) %>%
    mutate(Intercept = `(Intercept)`) %>%
    mutate(Experiment = unique(orig_data$experiment)) %>%
    select(c(Intercept, Subject, Experiment))
  rownames(data) <- NULL
  data
}

# Custom function to calculate correlations with adjusted p-values in ggpairs
cor_with_p_adjust <- function(data, mapping, method = "pearson", ...) {
  # Extract x and y variables
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)
  
  # Perform correlation test
  test <- cor.test(x, y, method = method)
  
  # Extract p-value and adjust using Bonferroni correction
  p_value <- test$p.value
  # Bonferroni correction: number of comparisons is choose(n, 2)
  p_value_adj <- p.adjust(p_value, method = "BH", n = choose(ncol(data), 2))
  # “holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”
  
  # replace = 1 with ≈ 1
  # Determine the p-value label
  if (p_value_adj >= 0.995) {
    p_label <- "≈1"
  } else {
    p_label <- format.pval(p_value_adj, digits = 2)
    p_label <- paste0("=", p_label)
  }
  
  # Create a label for ggally_text
  #label <- paste("r = ", round(test$estimate, 2), "\n", "p = ", format.pval(p_value_adj, digits = 2))
  label <- paste0("r=", round(test$estimate, 2), "\n", "p", p_label)
  
  
  # Create ggally_text object
  #ggally_text(label = label, color = ifelse(p_value_adj < 0.05, "red", "black"), ...)
  # also remove grid lines
  ggally_text(label = label, color = ifelse(p_value_adj < 0.05, "red", "black"), ...) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())  # Remove gridlines
}
