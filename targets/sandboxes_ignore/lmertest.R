library(lmerTest)

packages <- c("dplyr", "ggplot2", "ggsignif", "readr", "lmerTest", "emmeans", "magrittr", "ggpubr", "data.table",
  "tidyverse", "tidyquant", "ggdist", "ggthemes", "broom", "dplyr", "purrr", "rstatix", "tidyr")
lapply(packages, require, character.only = TRUE)


data <- get_preprocess_data("eegnet.csv")


# 
data <- tar_read(data_eegnet)
#
model <- lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + experiment + (hpf + lpf + emc + mac + base + det + ar + experiment | subject) ",
     control = lmerControl(calc.derivs = FALSE),
     data = data)

data2 <- data %>% filter(experiment=="N170")

# faster according to https://www.reddit.com/r/rstats/comments/t17t8a/lmer_model_takes_2_days_to_converge_is_it_a_bad/
model <- lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + experiment + (hpf + lpf + emc + mac + base + det + ar + experiment | subject) ",
              control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
              data = data)


# simulations
?lmer


# for each variable
#   for each subject
#     for 1000 iters
#       shuffle factors
#       hlm
#       extract p value
#     compare p values with actual p value


set.seed(42)
slice_sample()