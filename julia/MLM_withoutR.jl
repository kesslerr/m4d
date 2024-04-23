
# navigate to dir
cd("/Users/roman/GitHub/m4d/julia/")

# import libraries
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

#randint = ARGS[1]
#interact = ARGS[2]

# DEBUG
randint = 4610
interact = "true"

# Convert the value of randint to a string, so the R function below can use it
randint_str = string(randint)
interact = parse(Bool, interact)

# define column type so that the strings (in experiment) is long enough
column_types = Dict(
    "experiment" => String31,  # Assuming you have a column named "experiment" in your CSV file
    # Add more column types if needed
)

# import dataframe
#data = CSV.read(string("data_", randint, ".csv"), DataFrame, types=column_types) 
data = CSV.read(string("../targets/eegnet.csv"), DataFrame, types=column_types) 
#data = CSV.read("data_eegnet.csv", DataFrame) 

data = filter(row -> row.experiment == "ERN", data)
select!(data, Not(:experiment, :dataset))


# define factor levels
#describe(data)



# convert to factors
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
#data[!, :dataset] = categorical(data[!, :dataset])

# mixed model create formula
if length(unique(data[!, :experiment])) > 1
    formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment + ( ref + hpf + lpf + emc + mac + base + det + ar + experiment | subject))
else
    # fpr single experiments
    if !interact
        formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject))
    elseif interact
        #formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + ( (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 | subject))
        # to remove the correlation between the random effects --> speeds up processing time tremendously for high number of rfx
        formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + zerocorr( (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 | subject))
        # DEBUG, make it faster for debuuging by skipping some vars
        #formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + ar) ^ 2 + zerocorr( (ref + hpf + lpf + emc + mac + base + ar) ^ 2 | subject))
    end
end


# selected interaction subset
formula = @formula(accuracy ~ (hpf + lpf + base + det) ^ 2 + (hpf + lpf + emc + mac + ar ) ^2 + ref*hpf + ref*lpf + ref*ar +
                    zerocorr( (hpf + lpf + base + det) ^ 2 + (hpf + lpf + emc + mac + ar ) ^2 + ref*hpf + ref*lpf + ref*ar | subject) )

#formula = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + zerocorr( (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 | subject))



# fit it
model = fit(LinearMixedModel, formula, data) # suppress output into R


# convert to R
rmodel = (model, data);
#R"install.packages('afex', type = 'source')"
#R"library('afex')" # needed for converting an lmer model with zerocorr

elapsed_time = @elapsed begin
    #@rput rmodel
    @rput rmodel
end



# export as file for import in R
R"saveRDS(rmodel, file = paste0('model_',$randint_str,'.rds'))" # the $ brings the julia variable into R

# timing
# 4 vars: 128s --> 2 min (ref + hpf + lpf + ar)
# 5 vars: 300s --> 5 min (ref + hpf + lpf + emc + ar)
# 6 vars: 590s --> 10 min       + mac
# 7 vars: 1400s --> 23 min      + base

# R"save(model, file = 'model.rds')"

# at least 4:30 but still running, objective -204330, 20000 iterations 1s / iter
# can the model be speed up? https://juliastats.org/MixedModels.jl/dev/constructors/
# taking correlations between rfx out already improves from >>4:30h to 0:005h

# error "Error in initializePtr() : function 'cholmod_factor_ldetA' not provided by package 'Matrix'"
# https://stackoverflow.com/questions/77481539/error-in-initializeptr-function-cholmod-factor-ldeta-not-provided-by-pack
# uninstall Matrix and afex, and reinstall from source

# r squared of the model
# using Statistics
# y_actual = data[!, :accuracy]
# y_predicted = predict(model)
# r2 = 1 - sum((y_actual .- y_predicted).^2) / sum((y_actual .- mean(y_actual)).^2)


# formula2 = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject))
# model2 = fit(LinearMixedModel, formula2, data) # suppress output into R
# y_predicted2 = predict(model2)
# r22 = 1 - sum((y_actual .- y_predicted2).^2) / sum((y_actual .- mean(y_actual)).^2)