
# test interactions

modeli2 <- tar_read(sliding_LMi2_exp, branches=1)[[1]]
model <- tar_read(sliding_LM_exp, branches=1)[[1]]

models <- summary(model)
modeli2s <- summary(modeli2)

models$adj.r.squared
modeli2s$adj.r.squared