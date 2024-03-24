# heatmaps
library(colorspace)
library(ggpattern)
# TODO, concatenate the data in the pipeline

data = data.frame()
i <- 0
for (experiment in c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")){
  i <- i + 1
  tmp <- tar_read(eegnet_HLM_exp_emm_means, branches=i)[[1]]
  # normalize to zero for each experiment to have it centered and in a similar range
  tmp$emmean <- tmp$emmean / mean(tmp$emmean) - 1
  tmp[["experiment"]] <- experiment
  data <- rbind(data, tmp)
}

data <- tar_read(eegnet_HLM_emm_means_comb) %>% 
  filter(!(experiment == "ALL" & variable == "experiment")) %>% # delete the experiment compairson in the full data
  group_by(experiment) %>%   
  mutate(emmean = emmean / mean(emmean) - 1)

data$significance <- "No"
data$significance[1:10] <- "Yes"
# temporary make some patterns in some fields



# PLOT

ggplot(data, aes(y = 0, x = level, fill = emmean)) +
  geom_tile() +
  # in aes: , pattern=significance
  # geom_tile_pattern(pattern_color = NA,
  #                   pattern_fill = "green",
  #                   pattern_angle = 45,
  #                   pattern_density = 0.25,
  #                   pattern_spacing = 0.025,
  #                   pattern_key_scale_factor = 1) +
  # scale_pattern_manual(values = c(Yes = "circle", No = "none")) +
  # guides(pattern = guide_legend(override.aes = list(fill = "white"))) +
  # 
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


# TOOD: maybe black box for each facet that has any significant effect?
# TODO: add hlm_all as extra experiment




set.seed(40)
df2 <- data.frame(Row = rep(1:9,times=9), Column = rep(1:9,each=9),
                  Evaporation = runif(81,50,100),
                  TreeCover = sample(c("Yes", "No"), 81, prob = c(0.3,0.7), replace = TRUE))

ggplot(data=df2, aes(x=as.factor(Row), y=as.factor(Column),
                     pattern = TreeCover, fill= Evaporation)) +
  geom_tile_pattern(pattern_color = NA,
                    pattern_fill = "black",
                    pattern_angle = 45,
                    pattern_density = 0.5,
                    pattern_spacing = 0.025,
                    pattern_key_scale_factor = 1) +
  scale_pattern_manual(values = c(Yes = "circle", No = "none")) +

  scale_fill_gradient(low="#0066CC", high="#FF8C00") +
  coord_equal() + 
  labs(x = "Row",y = "Column") + 
  guides(pattern = guide_legend(override.aes = list(fill = "white")))