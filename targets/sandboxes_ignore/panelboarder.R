library(ggplot2)
library(dplyr)
library(colorspace)

# individualize panel border


data <- tar_read(eegnet_HLM_emm_means_comb)

data <- data %>% 
  # Apply replacements batchwise across all columns
  mutate(variable = recode(variable, !!!replacements)) %>%
  # delete the experiment compairson in the full data
  filter(!(experiment == "ALL" & variable == "experiment")) %>% 
  # center around zero for better comparability
  group_by(experiment) %>%
  mutate(emmean = (emmean / mean(emmean) - 1) * 100 ) # now it is percent

data <- data %>% 
  mutate(mark_facet = ifelse(experiment == "ALL","yes","no")) 

# GGPATTERN

# PLOT

ggplot(data, aes(y = 0, x = level, fill = emmean, pattern=mark_facet)) +
  geom_tile() +
  # in aes: , pattern=significance
  geom_tile_pattern(pattern_color = NA,
                    pattern       = "placeholder", #'magick',
                    pattern_type  = "kitten",
                    pattern_scale = 0.5,
                    pattern_fill = "black",
                    pattern_angle = 0,
                    #pattern_density = 0.25,
                    #pattern_spacing = 0.025,
                    pattern_key_scale_factor = 1) +
  #scale_pattern_manual(values = c("yes" = "circle", "no" = "none")) +
  guides(pattern = guide_legend(override.aes = list(fill = "white"))) +

  facet_grid(experiment~variable, scales="free") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  #scale_fill_continuous_diverging(palette = "Purple-Green") +
  #scale_fill_continuous_diverging() +
  scale_fill_continuous_diverging(palette = "Blue-Red 3", 
                                  l1 = 45, # luminance at endpoints
                                  l2 = 100, # luminance at midpoints
                                  p1 = .9, 
                                  p2 = 1.2) +
  labs(x="processing level",
       y="",
       fill="relative\nmarginal\nmeans")




















ggplot(data, aes(y = 0, x = level, fill = emmean)) +
  geom_tile() +
  geom_point(aes(alpha=mark_facet), shape=8, size=2) +
  facet_grid(experiment~variable, scales="free") +
  # theme(axis.text.y = element_blank(),
  #       axis.ticks.y = element_blank(),
  #       #panel.border = element_rect(colour = "lightgrey", fill=NA, size=5)
  #       #panel.border = element_blank(),      # Remove borders from all facets
  #       #strip.background = element_rect(fill = "grey", color = NA)  # Set background color for facet labels
  #       ) +
  scale_fill_continuous_diverging(palette = "Blue-Red 3", 
                                  l1 = 45, # luminance at endpoints
                                  l2 = 100, # luminance at midpoints
                                  p1 = .9, 
                                  p2 = 1.2) +
  labs(x="processing level",
       y="",
       fill="delta\nfrom\nmarginal\nmean\n(%)") 

#GEMINI



ggplot(data, aes(y = 0, x = level, fill = emmean)) +
  geom_tile() +
  facet_grid(experiment~variable, scales="free") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        # ... other theme options
  ) +
  scale_fill_continuous_diverging(palette = "Blue-Red 3", 
                                  l1 = 45, l2 = 100, p1 = .9, p2 = 1.2) +
  labs(x="processing level",
       y="",
       fill="delta\nfrom\nmarginal\nmean\n(%)",
       facet.label = paste(ggplot2::labeller(facet.name = label), 
                           ifelse(data$experiment == "ALL", " **(black border)**", ""), sep = ""))









# ChatGPT
# Convert 'significance' to factor if it's not already
data2$significance <- factor(data2$significance)

max_level <- max(as.numeric(as.character(data2$level)))  


ggplot(data2, aes(y = 0, x = level, fill = emmean)) +
  geom_tile() + 
  facet_grid(experiment ~ variable, scales = "free") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  scale_fill_continuous_diverging(palette = "Blue-Red 3", l1 = 45, l2 = 100, p1 = 0.9, p2 = 1.2) +
  labs(x = "processing level",
       y = "",
       fill = "delta\nfrom\nmarginal\nmean\n(%)") #+
  #geom_rect(aes(xmin = "average", xmax = "P9P10", 
  #              ymin = -0.5, ymax = 0.5, 
  #              color = significance, fill = NA)) 



outline <- data.frame(
  cyl = c(4, 6, 8), 
  outline_color = c('green', 'orange', 'red')   
)






# from: 

# Outline colours 
outline <- data.frame( 
  cyl = c(4, 6, 8), 
  outline_color = c('green', 'orange', 'red') 
) 

# Points defining square region for background 
square <- with(mtcars, data.frame( # TODO what does WITH do?
  x = c(-Inf, Inf, Inf, -Inf), 
  y = c(-Inf, -Inf, Inf, Inf)
))

ggplot(mtcars, aes(x = mpg, y = wt)) + 
  geom_polygon(aes(x = x,y = y, color = outline_color, fill = NA), data = merge(outline, square)) + 
  geom_point() + 
  scale_fill_identity() + 
  facet_grid(. ~ cyl) 