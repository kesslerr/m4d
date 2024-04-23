
# compare with and without interactions

using Parsers # for getting input arguments
using DataFrames
using CSV
using Plots
using MixedModels

# https://github.com/palday/JellyMe4.jl
ENV["LMER"] = "afex::lmer_alt" # set before import RCall and JellyMe4 to be able to convert zerocorr(rfx) correctly
using RCall
using JellyMe4

using RData
using CategoricalArrays # for categorical 
using GLM # the r2 function is in here
using Statistics # cor

# data must already exist
data = CSV.read("../targets/eegnet.csv", DataFrame)
data = filter(row -> row.dataset == "ERPCORE", data)
data = select!(data, Not(:dataset))

# Create arrays to store results
interact_results = Vector{Bool}()
experiment_results = Vector{String}()
r2_results = Vector{Float64}()

for interact in [true, false]
    println("interact = ", interact)
    
    # Loop over unique values of data$experiment
    for experiment in unique(data.experiment)
        println("experiment = ", experiment)

        # filter the data
        filtered_data = filter(row -> row.experiment == experiment, data)
        
        # formula
        if !interact
            formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject))
        elseif interact
            formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + zerocorr( (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 | subject))
        end
        # fit
        model = fit(LinearMixedModel, formula, filtered_data)

        # R2 --> TODO cite: https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#how-do-i-compute-a-coefficient-of-determination-r2-or-an-analogue-for-glmms
        #r_squared = r2(model)

        predictions = predict(model, filtered_data)
        r_squared = cor(predictions, filtered_data.accuracy)^2

        push!(interact_results, interact)
        push!(experiment_results, experiment)
        push!(r2_results, r_squared)
    
    end
end


# Create DataFrame from results
results_df = DataFrame(
    interact = interact_results,
    experiment = experiment_results,
    r2 = r2_results
)