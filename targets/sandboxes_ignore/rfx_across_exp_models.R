
# similar RFX across experiments?


model <- tar_read(eegnet_HLM_exp, branches=1)[[1]] # TODO: use pattern, and then combine the results

data <- ranef(model)$subject %>%
  mutate(Subject = rownames(.)) %>%
  mutate(Intercept = `(Intercept)`) %>%
  select(c(Intercept, subject))
rownames(data) <- NULL
