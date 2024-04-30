
# JULIA Debug according to https://github.com/palday/JellyMe4.jl/issues/51
library(JuliaCall)
options(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")
julia_install_package("StatsModels") 

julia_library("MixedModels")
julia_library("RCall")
julia_library("DataFrames")
julia_library("StatsModels")
julia_assign("dat",nlme::Machines)

julia_command("my_mod = fit(MixedModel, @formula(score ~ 1 + Machine + (1 + Machine|Worker)),dat)")
julia_library("JellyMe4")

julia_eval("(my_mod,dat)")

julia_eval("RCall.Const.GlobalEnv[:m_machines] = robject(:my_mod, m_machines)") #https://github.com/palday/JellyMe4.jl/issues/72

####
julia_command("my_mod = fit(MixedModel, @formula(score ~ 1 + Machine + (1 + Machine|Worker)),dat)")
julia_library("JellyMe4")
julia_command("return_mod = (my_mod, dat)")

# this would actually work, if matrix package would have the problem
#https://github.com/palday/JellyMe4.jl/issues/72
julia_command("RCall.Const.GlobalEnv[:return_mod] = robject(:lmerMod, return_mod)")

# Pass as a tuple
julia_eval("(my_mod, dat)") 






# package versions
julia_command("using Pkg; Pkg.status()")

# update all packages
julia_command("using Pkg; Pkg.update()")


#### SEPARATE JULIA FILE

# export dataframe for it
data <- tar_read(data_eegnet) %>% filter(experiment=="ERN") %>% select(-c("experiment"))
data %>% write.csv("../julia/data.csv", row.names = FALSE)

# HERE IN JULIA

# import converted model
load("../julia/model.rds")
library(tidyr)
library(gtools)
variables = c("ref", "hpf","lpf","emc","mac","base","det","ar")
em <- est_emm(return_model, variables)

em2 <- tar_read(eegnet_HLM_exp_emm, branches=1)[[1]]



## julia system2

# Define the path to your Julia executable
julia_executable <- "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/julia" # maybe without julia

# Define the path to your Julia script
julia_script <- "/Users/roman/GitHub/m4d/julia/MLM.jl"

# Run the Julia script using system2()
system2(command = julia_executable, args = c(julia_script))

