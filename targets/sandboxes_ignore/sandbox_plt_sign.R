
## try plot with significances

data <- tar_read(marginal_means)
sign <- tar_read(stats_all)

library(ggplot2)
library(ggdist)
library(ggsignif)

#mean_accuracy <- aggregate(accuracy ~ experiment, data, mean)

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
  facet_wrap(~variable, scales="free")
  
  # Add significance indicators
  #geom_signif(comparisons = list(c("200ms", "400ms")), #list(c("A", "B")), 
  #            map_signif_level = TRUE, 
  #            textsize = 3,
  #            step_increase = 0.05)

  




# Apply the function to each subset of the data based on the levels of the variable
plots <- lapply(split(data, data$variable), compute_significance)

# Plot the results in a grid using gridExtra
library(gridExtra)

grid.arrange(grobs = plots, ncol = 2)  # Adjust ncol as needed





library(ggplot2)
library(ggdist)  # for stat_halfeye
library(ggsignif)  # for geom_signif
library(gridExtra)
library(dplyr)  # for pipe operator and group_by
library(tidyr)  # for crossing function


# Assuming your dataframe is named data

# Create a function to compute significance tests for each facet
compute_significance <- function(df) {
  combinations <- data %>%
    distinct(factor) %>%
    pull() %>%
    crossing(factor1 = ., factor2 = .) %>%
    filter(factor1 != factor2) %>%
    mutate(combinations = list(c(factor1, factor2)))
  
  ggplot(df, aes(x = factor, y = accuracy)) +
    stat_halfeye(
      adjust = 0.5,
      justification = -0.2,
      .width = 0,
      point_colour = NA,
      scale = 0.5
    ) +
    geom_boxplot(
      width = 0.12,
      outlier.color = NA,
      alpha = 0.5
    ) +
    stat_dots(
      side = "left",
      justification = 1.1,
      binwidth = 0.005,
    )# +
    #geom_signif(comparisons = combinations$combinations,  # Replace with appropriate comparisons
    #            map_signif_level = TRUE, 
    #            textsize = 3,
    #            step_increase = 0.05) +
    theme_bw()  # Add your preferred theme here
}

# Apply the function to each subset of the data based on the levels of the variable
plots <- lapply(split(data, data$variable), compute_significance)

# Plot the results in a grid using gridExtra
library(gridExtra)

grid.arrange(grobs = plots, ncol = 3)  # Adjust ncol as needed



# test, if loop through wrap works

data <- data.frame(
  x = rep(1:10, 3),
  y = rnorm(30),
  facet = rep(letters[1:3], each = 10),
  group = rep(c("A", "B"), each = 15)
)

# Create a base plot
base_plot <- ggplot(data, aes(x = x, y = y)) +
  geom_point() +
  facet_wrap(~variable, scales = "free")

# Loop through each facet and add geom_signif
final_plot <- lapply(unique(data$variable), function(f) {
  facet_data <- subset(data, variable == f)
  p <- base_plot %+% facet_data
  
  # Add geom_signif to the facet plot
  p + geom_signif(data = facet_data, aes(xmin = 5, xmax = 10, annotations = c("***")), 
                  manual = TRUE, 
                  step_increase = 0.05,
                  textsize = 3)
})

# Print or display the final plots
print(final_plot)




# paired
library(ggpubr)
library(combinat)


ggpaired(filter(data, variable != "experiment"), 
         x = "factor", 
         y = "accuracy",
         id = "subject",
         line.color = "gray",
         line.size = 0.6
         #facet.by = c("variable")
         ) + 
  facet_grid(.~variable, scales="free") #+
  #facet_wrap(~variable, scales="free") #+
#stat_compare_means(paired = TRUE,
  #                   comparisons = list(c(FALSE, TRUE)))

  
