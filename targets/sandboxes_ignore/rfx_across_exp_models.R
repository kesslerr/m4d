
# similar RFX across experiments?


model <- tar_read(eegnet_HLM_exp, branches=1)[[1]] # TODO: use pattern, and then combine the results
orig_data <- tar_read(data_eegnet_exp, branches=1) %>% filter(tar_group==1) # to get the EXP information

# start here


data <- ranef(model)$subject %>%
  mutate(Subject = rownames(.)) %>%
  mutate(Intercept = `(Intercept)`) %>%
  mutate(Experiment = unique(orig_data$experiment)) %>%
  select(c(Intercept, Subject, Experiment))
rownames(data) <- NULL

data <- tar_read(rfx_exp)

ggpaired(data, x = "Experiment", y = "Intercept",
         color = "Subject", 
         line.color = "gray", line.size = 0.4,
         palette = "jco")


# maybe pairwise corr plots?
# therefore, i need a wide plot, not long
wide_data <- data %>% 
  pivot_wider(names_from = Experiment, values_from = "Intercept") %>% 
  select(-c("Subject")) # remove sub for now

ggpairs(wide_data)


ggpaired(
  data,
  cond1,
  cond2,
  x = NULL,
  y = NULL,
  id = NULL,
  color = "black",
  fill = "white",
  palette = NULL,
  width = 0.5,
  point.size = 1.2,
  line.size = 0.5,
  line.color = "black",
  linetype = "solid",
  title = NULL,
  xlab = "Condition",
  ylab = "Value",
  facet.by = NULL,
  panel.labs = NULL,
  short.panel.labs = TRUE,
  label = NULL,
  font.label = list(size = 11, color = "black"),
  label.select = NULL,
  repel = FALSE,
  label.rectangle = FALSE,
  ggtheme = theme_pubr(),
)




ggplot(data, aes(x = Experiment, y = Intercept, color=Subject)) +
  geom_line()


+
  scale_fill_gradient2() + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 