
# sandbox
source("R/functions.R")
library(targets)
library(emmeans)
library(coin)

library(ggplot2)
library(magrittr)
library(ggsignif)

hlm <- tar_read(hlm_all_experiments)
summary(hlm)


#ref + hpf + lpf + base + det + ar + emc + mac + experiment

variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment")
variables = c("experiment")
for (variable in variables){
  # MAIN EFFECTS (1 factor)
  emm_var <- emmeans(hlm, specs = formula(paste0(c("pairwise ~ ",variable)))) # ref, within exp
  #print(emm)
  
  # convert results to dfs
  df <- emm_var$emmeans %>% as.data.frame()
  con <- emm_var$contrasts %>% as.data.frame()
  
  #sign_list <- find_significant_combinations(df, con)
  sign_list <- find_significant_combinations(df[[variable]], con)
  
  # Create the plot
  p1 <- ggplot(df, aes_string(x = variable, y = "emmean")) +
    geom_bar(stat = "identity", fill = "grey") +  # Bar plot with emmean values
    geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.2) +  # Add error bars
    labs(x = variable, y = "accuracy") +  # Add labels
    # horizontal bar at 0.5
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +  # Add horizontal line at y = 0.5
    theme_light() +  # Apply minimal theme
    # significance values
    #geom_signif(comparisons = list(c("ERN", "N170"), c("ERN", "MMN")), annotations = c("***"), textsize = 3, vjust = 0.5, map_signif_level = TRUE, step_increase=0.04) +  # Add significance bars between ERN and N170
    geom_signif(comparisons = sign_list, annotations = c(""), #annotations = c("***"), 
                textsize = 3, vjust = 0.5, map_signif_level = TRUE, step_increase=0.04,
                y_position = 0.05,
                tip_length=0) +  # Add significance bars between ERN and N170
    
    coord_cartesian(ylim = c(0,1)) #c(0.45, max(df$emmean) + length(sign_list) * 0.02))  # Set x-axis limits, max(df$emmean # TODO: find better formula for the heights
  
  plot(p1)
}

# pairwise significances from other table
# TODO: make it more versatile

# input: features = df$experiment, combinatinos = con$contrast


# TODO: sample 5% of the total data and see statistical significance?
# TODO: or, average across accs of all pipelines (per sub), 


# TEST: can I 20x copy my data, will everything become more significant? This might indicate, that HLM with mutliverse does inflate sign.
data <- tar_read(factorized_data)
replicated_data <- replicate(50, data, simplify = FALSE)
library(dplyr)
concatenated_data <- bind_rows(replicated_data)

hlm50 <- lmer(data=concatenated_data, formula="accuracy ~ ref + hpf + lpf + base + det + ar + emc + mac + experiment + (1 | subject)")

summary(hlm10)

summary(hlm)


# TEST: can I divide my df / 10 and everything becomes less sign? This might indicate, that HLM with mutliverse does inflate sign.
data <- tar_read(factorized_data)
reduced_data <- data %>% filter(hpf == "None", lpf == "None", base == "200ms", det == "offset", emc=="None", mac=="None", ref=="average")

hlmRed <- lmer(data=reduced_data, formula="accuracy ~  ar +  experiment + (1 | subject)") #ref +emc + mac +
summary(hlmRed)
summary(hlm10)

summary(hlm)


# all interactions 
data_ERN <- data %>% filter(experiment=="ERN")
hlm_ERN <- lmer(data=data_ERN, formula="accuracy ~ ref * hpf * lpf * base * det * ar * emc * mac + (1 | subject)")



# Average across factors, then do permutation test

variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment")
for (variable in variables){
  avg <- data %>% 
  group_by(subject, !!sym(variable)) %>%
  summarize(average_accuracy = mean(accuracy, na.rm = TRUE)) %>%
  as.data.frame()
  #print(avg)
  # for each factor, across all subs, calculate paired sample permut test
}

library(broman)
avg$subject <- as.factor(avg$subject)

# Perform the paired permutation test
f1 <- avg %>% filter(experiment=="N170") %>% select(average_accuracy)
f2 <- avg %>% filter(experiment=="P3") %>% select(average_accuracy)

t.test(f1$average_accuracy, f2$average_accuracy, paired=TRUE)

paired.perm.test(f1$average_accuracy- f2$average_accuracy, n.perm = 50000, pval = TRUE) # n.perm = NULL for all permutations



###


library(tidyr)


variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment")

# Create list to store marginal means for each variable
marginal_means_list <- list()

# Loop over each variable and estimate marginal means for the main effect of that variable
for (variable in variables) {
  # Create formula for the main effect of the current variable
  formula <- as.formula(paste("~", variable, "| subject"))
  
  # Estimate marginal means for the main effect of the current variable
  marginal_means <- emmeans(hlm, formula) %>% as.data.frame()
  
  # Assuming your dataframe is named `combined_df`
  long_df <- marginal_means %>%
    pivot_longer(cols = c(variable),
                 names_to = "variable",
                 values_to = "factor")
  # Store marginal means in the list
  marginal_means_list[[variable]] <- long_df
  
}

concatenated_df <- do.call(rbind, marginal_means_list)


p <- ggplot(concatenated_df, aes(x = factor, y = emmean)) +
  geom_bar(stat = "identity", position = "dodge", fill = "grey") +
  facet_wrap(~ variable, scales = "free") +  # Facet by the variable column
  labs(x = "Factor", y = "Emmean") +         # Labels for x and y axes
  theme_bw()                                # Use a white-background theme
  #scale_y_continuous(limits = c(0.5, NA)) 

# Print the plot
print(p)


# Create the point plot with error bars and facets for each level of "variable"
p <- ggplot(concatenated_df, aes(x = factor, y = emmean, 
                                # ymin = asymp.LCL, ymax = asymp.UCL)) +
                                 ymin = emmean - SE, ymax = emmean + SE)) +
  geom_point(position = position_dodge(width = 0.2), size = 3) +
  geom_errorbar(position = position_dodge(width = 0.2), width = 0.2) +
  facet_wrap(~ variable, scales = "free_y") +  # Facet by the variable column with free y-axis scales
  labs(x = "Factor", y = "Emmean") +           # Labels for x and y axes
  theme_bw()                                 # Use a white-background theme
  #scale_y_continuous(limits = c(0.5, NA))      # Set y-axis limits starting from 0.5

# Print the plot
print(p)



library(tidyverse)
library(dplyr)
library(tidyquant)
library(ggdist)
library(ggthemes)

# Assuming your data frame is called df
df <- data
variables = c("ref","hpf","lpf","base","det","ar","emc","mac","experiment")
#variables = c("ref")

# Calculate average accuracy by levels of columns from "ref" to "experiment" by subject
dfs_list <- list()
for (variable in variables) {
  average_df <- df %>%
    group_by(subject, !!sym(variable)) %>%
    summarize(accuracy = mean(accuracy))# %>%
    #ungroup()
  average_df$variable <- names(average_df)[2]
  average_df$factor <- average_df[[variable]]
  average_df[[variable]] <- NULL
  dfs_list <- append(dfs_list, list(average_df))
}
# Concatenate the data frames in the list
concatenated_df <- bind_rows(dfs_list)



### PLOT EMMs

# https://rpubs.com/rana2hin/raincloud
ggplot(data = concatenated_df, aes(x = factor, y = accuracy)) +
  
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
  facet_wrap(~variable, scales="free")


