
# backwards regression or so

using Parsers # for getting input arguments
using DataFrames
using CSV
using Plots
using MixedModels
using RCall
using JellyMe4
using RData
using CategoricalArrays # for categorical 
using GLM # the r2 function is in here
using Statistics # cor
using Effects, GLM, StatsModels, StableRNGs, StandardizedPredictors



data = CSV.read("../targets/eegnet.csv", DataFrame)
#only keep rows with experiment == "ERN"  (automatically drops MIPDB)
data = data[data.experiment .== "ERN", :]
# discard dataset column
select!(data, Not(:dataset))

levels_hpf = ["None", "0.1", "0.5"];
levels_lpf = ["None", "6", "20", "45"];
levels_ref = ["average", "Cz", "P9P10"];
levels_emc = ["None", "ica"];
levels_mac = ["None", "ica"];
levels_base = ["200ms", "400ms"];
levels_det = ["offset", "linear"];
levels_ar = [false, true];
levels_experiment = ["ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3"];


data[!, :hpf] = categorical(data[!, :hpf], levels=levels_hpf);
data[!, :lpf] = categorical(data[!, :lpf], levels=levels_lpf);
data[!, :ref] = categorical(data[!, :ref], levels=levels_ref);
data[!, :emc] = categorical(data[!, :emc], levels=levels_emc);
data[!, :mac] = categorical(data[!, :mac], levels=levels_mac);
data[!, :base] = categorical(data[!, :base], levels=levels_base);
data[!, :det] = categorical(data[!, :det], levels=levels_det);
data[!, :ar] = categorical(data[!, :ar], levels=levels_ar);
if "experiment" in names(data)
    data[!, :experiment] = categorical(data[!, :experiment], levels=levels_experiment);
end


#formula = @formula(accuracy ~ (ref + hpf + lpf + emc) ^ 2 + zerocorr( (ref + hpf + lpf + emc) ^ 2 | subject));
formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + zerocorr( (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 | subject));

model = fit(LinearMixedModel, formula, data) # suppress output into R

emmeans(model)


emp = empairs(model)

emm = empairs(model, dof=Inf)
# group by column emc and average accuracy
grouped_df = combine(groupby(emm, :emc), :accuracy => mean => :avg_accuracy)

emm = emmeans(model, ~ emc, eff_col="accuracy")


rng = StableRNG(42)
growthdata = DataFrame(; age=[13:20; 13:20],
                       sex=repeat(["male", "female"], inner=8),
                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+ randn(rng, 16))
model_scaled = lm(@formula(weight ~ 1 + sex * age), growthdata;
                  contrasts=Dict(:age => ZScore(), :sex => DummyCoding()))

emmeans(model_scaled)


# try within function

elapsed_time = @elapsed begin
    model = fit(LinearMixedModel, formula, data) # suppress output into R
end


function lmmf(data, formula)
    model = fit(LinearMixedModel, formula, data) # suppress output into R
    return model
end

elapsed_time = @elapsed begin
    model = lmmf(data, formula)
end


