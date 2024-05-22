
library(ggplot2)
library(ggsankey)
library(dplyr)
library(magrittr)
library(paletteer)

# own data
data <- tar_read(data_eegnet)
data %<>% filter(subject == "sub-001") %>%
  filter(experiment == "N170") %>%
  select(-c(subject, accuracy, experiment)) 

# now change the names of all columns with the replacements
names(data) <- recode(names(data), !!!replacements)

# make long
data_long <- data %>%
  make_long(names(data)) %>%
  mutate(node = recode(node, !!!replacements)) %>% # also replace with better names
  mutate(next_node = recode(next_node, !!!replacements))
  


ggplot(data_long, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
  geom_sankey(flow.alpha = .6,
              node.color = "gray20") +
  geom_sankey_label(size = 4, color = "white", fill = "gray40") +
  #scale_fill_viridis_d(drop = FALSE) +
  #paletteer::scale_fill_paletteer_d("colorBlindness::paletteMartin") +
  scale_fill_grey() +
  theme_sankey(base_size = 18) +
  labs(x = "processing step") +
  theme(legend.position = "none",
        plot.title = element_text(hjust = .5)) +
  ggtitle("Multiverse")