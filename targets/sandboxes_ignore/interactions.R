
# test interactions

#DEBUG
data = tar_read(data_eegnet_exp, branches=1) %>% filter(experiment == "ERN")
interactions = TRUE

# INFO: Julia should start its own process by each function call, so the ENV variables set should be individual
julia_library("Parsers, DataFrames, CSV, Plots, MixedModels, RData, CategoricalArrays")
if (interactions == TRUE){
  julia_command("ENV[\"LMER\"] = \"afex::lmer_alt\"") # set before import RCall and JellyMe4 to be able to convert zerocorr(rfx) correctly; https://github.com/palday/JellyMe4.jl (ReadMe)
} # caution, if zerocorr is used, now julia will automatically use afex::lmer_alt, and from then on, use lmer_alt for the remainder of the session
julia_library("RCall, JellyMe4")

julia_assign("data", data) # bring data into julia
if (interactions == FALSE){
  julia_command("formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + (ref + hpf + lpf + emc + mac + base + det + ar | subject));")
} else {
  julia_command("formula = @formula(accuracy ~ (ref + hpf + lpf) ^ 2 + zerocorr( (ref + hpf + lpf) ^ 2 | subject));") ## TODO all variables
}
julia_command("model = fit(LinearMixedModel, formula, data);")  # , verbose=false
julia_command("rmodel = (model, data);") # make it a tuple for conversion (Julia model doesn't have the data, but R model does); https://github.com/palday/JellyMe4.jl/issues/51, 
julia_command("RCall.Const.GlobalEnv[:rmodel] = robject(:lmerMod, rmodel);") # alternative to @rput; https://github.com/palday/JellyMe4.jl/issues/72
rmodel