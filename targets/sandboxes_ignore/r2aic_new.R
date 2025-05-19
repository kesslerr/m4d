
library(patchwork)


# sandbox r2/aic new plot version
# 
test <- tar_read(r2aic_table)


test2 <- test %>% filter(metric %in% c("R2", "AIC")) %>%
  # capitalize interactions
  mutate(interactions = ifelse(interactions == "false", "Absent", interactions)) %>%
  mutate(interactions = ifelse(interactions == "true", "Present", interactions))



# Prepare data for arrows: Match 'absent' to 'present' per facet & experiment
arrow_data <- test2 %>%
  filter(interactions %in% c("Absent", "Present")) %>%
  pivot_wider(names_from = interactions, values_from = value, 
              values_fn = list(value = first)) %>%
  rename(ymin = Absent, ymax = Present)  # Rename for clarity

# Plot with arrows
test2 %>%
  ggplot(aes(y = value, x = experiment, color = interactions)) +
  geom_point(size = 5) + 
  scale_color_grey(start = 0.2, end = 0.6) +
  # Arrows from 'absent' to 'present' per facet & experiment
  geom_segment(data = arrow_data, 
               aes(x = experiment, xend = experiment, 
                   y = ymin, yend = ymax), 
               arrow = arrow(type = "closed", length = unit(0.15, "inches")), 
               size = 0.8, inherit.aes = FALSE) +

  facet_wrap(metric ~ model, scales = "free_y", 
             labeller = labeller(metric = label_both)) +
  labs(y = "", x = "Experiment", fill = "Interactions") +
  theme_minimal()


# separate AIC and R2


test2 <- test %>% filter(metric %in% c("R2", "AIC")) %>%
  # capitalize interactions
  mutate(interactions = ifelse(interactions == "false", "Absent", interactions)) %>%
  mutate(interactions = ifelse(interactions == "true", "Present", interactions))
# Prepare data for arrows: Match 'absent' to 'present' per facet & experiment
arrow_data <- test2 %>%
  filter(interactions %in% c("Absent", "Present")) %>%
  pivot_wider(names_from = interactions, values_from = value, 
              values_fn = list(value = first)) %>%
  rename(ymin = Absent, ymax = Present)  # Rename for clarity

# AIC
p1 <- test2 %>%
  filter(metric== "AIC") %>%
  ggplot(aes(y = value, x = experiment, color = interactions)) +
  geom_point(size = 5) + 
  scale_color_grey(start = 0.2, end = 0.6) +
  # Arrows from 'absent' to 'present' per facet & experiment
  geom_segment(data = arrow_data %>% filter(metric == "AIC"), 
               aes(x = experiment, xend = experiment, 
                   y = ymin, yend = ymax), 
               arrow = arrow(type = "closed", length = unit(0.15, "inches")), 
               size = 0.8, inherit.aes = FALSE) +
  
  facet_wrap(. ~ model, scales = "free_y", 
             #labeller = labeller(metric = label_both)) +
  ) +
  labs(y = "AIC", x = "Experiment", color = "Interactions") +
  #theme_minimal() +
  theme(legend.position = c(0.9, 0.85))

p2 <- test2 %>%
  filter(metric== "R2") %>%
  ggplot(aes(y = value, x = experiment, color = interactions)) +
  geom_point(size = 5) + 
  scale_color_grey(start = 0.2, end = 0.6) +
  # Arrows from 'absent' to 'present' per facet & experiment
  geom_segment(data = arrow_data %>% filter(metric == "R2"), 
               aes(x = experiment, xend = experiment, 
                   y = ymin, yend = ymax), 
               arrow = arrow(type = "closed", length = unit(0.15, "inches")), 
               size = 0.8, inherit.aes = FALSE) +
  
  facet_wrap(. ~ model, scales = "fixed",  #fixed
             #labeller = labeller(metric = label_both)) +
  ) +
  labs(y = "R2", x = "Experiment", color = "Interactions") +
  #theme_minimal() +
  theme(legend.position = "none")


# merge both plots vertically
aicr2plot <- p1 + p2 +
  plot_layout(ncol = 2) +
  plot_annotation(title = "Model performance with and without interactions") +
  theme(plot.title = element_text(hjust = 0.5))

