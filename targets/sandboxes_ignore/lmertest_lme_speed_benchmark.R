
library(tictoc)
# speed test lme4 vs lmer

data <- tar_read(data_eegnet_exp, branches=1)

tic("lmerTest::lmer")
mod1 <- lmerTest::lmer(formula="accuracy ~ ref + ( ref | subject)",
               control = lmerControl(optimizer = "optimx", 
                                     calc.derivs = FALSE, 
                                     optCtrl = list(method = "nlminb", 
                                                    starttests = FALSE, kkt = FALSE)),
               data = data)

toc()
# 45 sec

tic("lme4::lmer")
mod2 <- lme4::lmer(formula="accuracy ~ ref + ( ref | subject)",
                       control = lmerControl(optimizer = "optimx", 
                                             calc.derivs = FALSE, 
                                             optCtrl = list(method = "nlminb", 
                                                            starttests = FALSE, kkt = FALSE)),
                       data = data)

toc()
# 33 sec


