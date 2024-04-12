
# JULIA tests
# https://hwborchers.github.io
library(dplyr)
library(JuliaCall)

#options(JULIA_HOME = "~/Programs/julia-1.8.5/bin/")
options(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")
julia_setup()

julia_call("sqrt", 2.0)

julia_command("a = sqrt(2.0)")
## [1] 1.4142135623730951

julia_command("a = sqrt(2.0);")   # no output printed

# to get the value, use julia_eval
a <- julia_eval("a = sqrt(2.0);")   # no output printed


# MLM

## JULIA
#using Pipe, DataFrames, StatFiles, GLM, MixedModels, Plots, StatsPlots, Statistics


# Julia within R
data <- tar_read(data_eegnet) %>% filter(experiment=="ERN") %>% select(-c("experiment"))

julia_library("DataFrames")
julia_library("RCall")
julia_library("MixedModels")
julia_library("JellyMe4")

julia_assign("data", data)

julia_command("model = fit(LinearMixedModel, @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)), data)")

julia_command("return_model = (model, data);")

julia_eval("return_model")
julia_command("@rput return_model")

# transfer it into R

#julia_command("model2 = (model, data);")
#julia_command("@rput model2;")

julia_command("thisModel = (model, data)")
julia_command("@rput thisModel;")


julia_eval("thisModel")

#model <- julia_eval("fit(LinearMixedModel, @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)), data)")
#julia_command("model = fit(LinearMixedModel, @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)), data)")


# https://github.com/palday/JellyMe4.jl/issues/51
# might be solvable!!
