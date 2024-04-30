
# try again and track time of model fitting in R instead of julia
# use || to keep it reasonable
library(dplyr)
#library(lme4)
library(tictoc)

data <- tar_read(data_eegnet) %>% filter(experiment=="ERN") %>% select(-c("experiment"))

tic()
model <- lme4::lmer(accuracy ~ ( ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + ( ( ref + hpf + lpf + emc + mac + base + det + ar ) ^ 2 || subject), 
                    #control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
                    control = lme4::lmerControl(optimizer = "nloptwrap2"), # glmerControl or lmerControl, depending; according to: https://stats.stackexchange.com/questions/132841/default-lme4-optimizer-requires-lots-of-iterations-for-high-dimensional-data
                    data = data)
toc()

# todo: test without this special wrapper
# 

#          
# fitting with glmer does not work --> deprecated
library(nloptr)
defaultControl <- list(algorithm="NLOPT_LN_BOBYQA",xtol_rel=1e-6,maxeval=1e5)
nloptwrap2 <- function(fn,par,lower,upper,control=list(),...) {
  for (n in names(defaultControl)) 
    if (is.null(control[[n]])) control[[n]] <- defaultControl[[n]]
  res <- nloptr(x0=par,eval_f=fn,lb=lower,ub=upper,opts=control,...)
  with(res,list(par=solution,
                fval=objective,
                feval=iterations,
                conv=if (status>0) 0 else status,
                message=message))
}


