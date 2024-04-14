
# ranef
library(tidyr)
library(ggplot2)

model <- tar_read(eegnet_HLM_exp, branches=1)[[1]]

data <- ranef(model)$subject

data_long <- data %>%
  pivot_longer(
    cols = names(.), #-c("subject"), #, # Select columns starting with "est"
    names_to = "level",         # Create the "level" column
    values_to = "mean" # Create the "conditional mean" column
  ) 

ggplot(data_long,
       aes(y=mean, x=level)) +
  geom_boxplot() +
  labs(y="Conditional Mean", x="Random Effect Term", title="ALL") +
  theme(axis.text.x = element_text(angle=90))

  #geom_density() +
  #facet_wrap(level~., scales = "free")


