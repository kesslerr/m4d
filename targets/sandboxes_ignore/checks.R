
# TODO
# other model checks (HLM)

# posterior predictive check
performance::check_model(hlmexp[[1]])


# model checks
plot(model)

# non-normality might / not must obscure statistical tests
qqnorm(resid(model))
qqline(resid(model))


qqplot.data(resid(model))


# Extract random effects at level 1
random_effects <- ranef(model, condVar = TRUE)$`subject`
random_effects_vector <- as.numeric(unlist(random_effects))
random_effects_vector <- random_effects_vector[!is.na(random_effects_vector)]
qqnorm(random_effects_vector)
qqline(random_effects_vector)

random_slopes <- names(ranef(model, condVar = TRUE))
# Create a Q-Q plot for each random slope
for (slope in random_slopes) {
  random_effects <- ranef(model, condVar = TRUE)[[slope]]
  random_effects_vector <- as.numeric(unlist(random_effects))
  random_effects_vector <- random_effects_vector[!is.na(random_effects_vector)]
  if (length(random_effects_vector) > 0) {
    qqnorm(random_effects_vector, main = paste("Q-Q Plot for Random Slope:", slope))
    qqline(random_effects_vector)
  }
}



# https://yjunechoe.github.io/posts/2020-06-07-correlation-parameter-mem/

# TODO find something from a tutorial

va <- VarCorr(model)
va <- as.data.frame(va)
vana <- va[!complete.cases(va), ]
vana$var1[is.na(vana$var1)] <- vana$grp[is.na(vana$var1)]
vana$grp = NULL
vana$var2 = NULL
names(vana)[1] <- "term"
vana$var_perc <- vana$vcov / sum(vana$vcov)

# collate all 

vana <- vana[order(vana$var_perc), ]


ggplot(vana, aes(x=term, y=var_perc)) +
  geom_bar(stat = "identity")