# table export

library(xtable)
library(dplyr)
library(tidyr)


## F TEST

data = tar_read(eegnet_HLM_emm_omni_comb)


# all experiments listed in one row as string concat
#df_new <- data %>%
#  filter(p.fdr < 0.5) %>% 
#  group_by(`model term`) %>%
#  summarize(significant_experiments = toString(experiment)) 


thisLabel <- "labeltest"
thisCaption <- "caption test"

# all in separate cols
output <- data %>%
  select(c(`model term`, experiment, p.fdr, sign.fdr)) %>%
  #filter(p.fdr < 0.05) %>% 
  select(-c(p.fdr)) %>%
  pivot_wider(
    names_from = experiment, 
    values_from = sign.fdr
  ) %>%
  mutate(across(everything(), ~ if_else(is.na(.x), "-", .x))) %>%
  xtable(type="latex",
         label=thisLabel,
         caption=thisCaption)


print(output, 
      #digits=5,
      file = paste0(manuscript_dir,"tables/eegnet_omni.tex"))


## Contrasts

data = tar_read(sliding_LM_emm_contrasts_comb)


# all in separate cols
output <- data %>%
  select(c(variable, level.1, level.2, experiment, significance)) %>%
  pivot_wider(
    names_from = experiment, 
    values_from = significance
  ) %>%
  mutate(across(everything(), ~ if_else(is.na(.x), "-", .x))) %>%
  xtable(type="latex",
         label=thisLabel,
         caption=thisCaption)

