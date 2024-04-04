
# test omnibus test (for each factor across all levels)

library(emmeans)

model <- tar_read(eegnet_HLM)
variables = c("ref", "hpf","lpf","base","det","ar","emc","mac","experiment")
fs <- data.frame()
for (variable in variables){
  emm <- emmeans(model, 
                 specs = formula(paste0(c("pairwise ~ ",variable))), 
                 #lmerTest.limit = 322560,
                 #pbkrtest.limit = 322560) # to not have inf df
  )
  f <- joint_tests(emm)
  fs <- rbind(fs, f)
}


print