
# diagnostics
library(gridExtra)
library(broom)
library(broom.mixed)
library(ggplot2)

fm1 <- tar_read(eegnet_HLM_exp, branches=1)[[1]]

data <- augment(fm1)

# Residuals vs. Fitted plot
p1 <- ggplot(data = data, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  labs(x = "Fitted Values", y = "Residuals") +
  theme_bw()

# Q-Q plot
p2 <- ggplot(data = data, aes(sample = .resid)) + 
  stat_qq() + 
  stat_qq_line() +
  labs(x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_bw()

# Scale-Location plot
p3 <- ggplot(data = data, aes(x = .fitted, y = sqrt(abs(.resid)))) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  labs(x = "Fitted Values", y = "sqrt(abs(Standardized Residuals))") +
  theme_bw()


grid.arrange(p1, p2, p3, ncol = 3) # Arrange in 3 columns









# residual vs fitted
plot(lmer_model)




plot(lmer_model, type = c("p", "smooth"))

# normalizty vs residuals
# are residuals normally distributed?
#library(lattice)
#qqmath(lmer_model)
# already done with other function

plot(lmer_model, sqrt(abs(resid(.))) ~ fitted(.))