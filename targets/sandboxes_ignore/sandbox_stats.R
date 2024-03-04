
# significance testing

data <- tar_read(marginal_means)

library(dplyr)
library(broom)
library(purrr)
library(lmerTest)

df_grouped <- data %>%
  group_by(variable)


# Perform paired t-test for each factor within each variable
#t_test_results <- df_grouped %>%
#  do(tidy(pairwise.t.test(.$accuracy, .$factor, p.adjust.method = "none"))) #"BY"

t_test_results <- df_grouped %>%
  #do(tidy(pairwise.wilcox.test(.$accuracy, .$factor, p.adjust.method = "none", paired=TRUE))) %>% #"BY"
  do(tidy(pairwise.t.test(.$accuracy, .$factor, p.adjust.method = "BY", paired=TRUE))) %>% #"BY"
  # Format p-values to decimal format
  mutate(p.value = format(p.value, scientific = FALSE)) #%>%
  # Extract test statistics from the pairwise.t.test output
  #mutate(test_statistic = map_dbl(data, ~.$statistic)) %>%
  # Select necessary columns
  #select(variable, comparison = group1, test_statistic, p.value)

# View the t-test results
print(t_test_results)



# with rstatix, as LM, then it returns stats and shold be same results
# https://stackoverflow.com/questions/77690426/pairwise-t-test-doesnt-return-statistic-and-estimate

library(rstatix)

df_exp = df_grouped %>% filter(variable=="hpf")

#rstatix::pairwise_t_test(df_exp, accuracy~factor, paired=TRUE, pool.sd=FALSE, detailed = TRUE)
rstatix::pairwise_t_test(df_exp, accuracy~factor, paired=TRUE, p.adjust.method = "BY", pool.sd=FALSE, detailed = TRUE)

# try for all variables in one
rstatix::pairwise_t_test(data, accuracy~factor | variable, paired=TRUE, p.adjust.method = "BY", pool.sd=FALSE, detailed = TRUE)


# with emmeans
mod1 <- lmer(data=df_exp , accuracy~factor + (1 | subject)) 
emmeans::emmeans(mod1, pairwise ~ factor, adjust="tukey", infer=c(TRUE, TRUE))$contrast



hlmall <- tar_read(hlm_all)

emm <- emmeans(hlmall, specs = formula(paste0(c("pairwise ~ ","hpf")))) # ref, within exp
emmdf <- emm$emmeans %>% as.data.frame() # leaving out contrasts for now
emmdfcon <- emm$contrasts %>% as.data.frame() # leaving out contrasts for now





## with all comparisons in once

# Load necessary libraries
library(dplyr)
library(rstatix)

# Assuming your dataframe is named df_exp

# Perform pairwise t-tests for each level of "variable"
test_results <- data %>%
  group_by(variable) %>%
  pairwise_t_test(accuracy ~ factor, paired = TRUE, p.adjust.method = "BY", pool.sd = FALSE, detailed = TRUE)

