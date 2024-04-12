
# navigate to dir
cd("/Users/roman/GitHub/m4d/julia/")

# import libraries
using Parsers # for getting input arguments
using DataFrames
using CSV
using Plots
using RCall
using MixedModels
using JellyMe4
using RData
using CategoricalArrays # for categorical 

randint = ARGS[1]

# DEBUG
#randint = 3760

# Convert the value of randint to a string, so the R function below can use it
randint_str = string(randint)


# define column type so that the strings (in experiment) is long enough
column_types = Dict(
    "experiment" => String7,  # Assuming you have a column named "experiment" in your CSV file
    # Add more column types if needed
)

# import dataframe
data = CSV.read(string("data_", randint, ".csv"), DataFrame, types=column_types) 
#data = CSV.read("data_eegnet.csv", DataFrame) 

# define factor levels
#describe(data)

# convert to factors
levels_hpf = ["None", "0.1", "0.5"]
levels_lpf = ["None", "6", "20", "45"]
levels_ref = ["average", "Cz", "P9P10"]
levels_emc = ["None", "ica"]
levels_mac = ["None", "ica"]
levels_base = ["200ms", "400ms"]
levels_det = ["offset", "linear"]
levels_ar = [false, true]
levels_experiment = ["ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3"]


data[!, :hpf] = categorical(data[!, :hpf], levels=levels_hpf)
data[!, :lpf] = categorical(data[!, :lpf], levels=levels_lpf)
data[!, :ref] = categorical(data[!, :ref], levels=levels_ref)
data[!, :emc] = categorical(data[!, :emc], levels=levels_emc)
data[!, :mac] = categorical(data[!, :mac], levels=levels_mac)
data[!, :base] = categorical(data[!, :base], levels=levels_base)
data[!, :det] = categorical(data[!, :det], levels=levels_det)
data[!, :ar] = categorical(data[!, :ar], levels=levels_ar)
if "experiment" in names(data)
    data[!, :experiment] = categorical(data[!, :experiment], levels=levels_experiment)
end
#data[!, :dataset] = categorical(data[!, :dataset])

# mixed model
if length(unique(data[!, :experiment])) > 1

    # for across experiments
    if "accuracy" in names(data)
        model = fit(LinearMixedModel, @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment + ( ref + hpf + lpf + emc + mac + base + det + ar + experiment | subject)), data);
    elseif "tsum" in names(data)
        model = fit(LinearMixedModel, @formula(tsum ~ ref + hpf + lpf + emc + mac + base + det + ar + experiment + ( ref + hpf + lpf + emc + mac + base + det + ar + experiment | subject)), data);
    else
        throw(ArgumentError("Neither accuracy nor tsum in data."))
    end
        
else

    # fpr single experiments
    if "accuracy" in names(data)
        model = fit(LinearMixedModel, @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)), data);
    elseif "tsum" in names(data)
        model = fit(LinearMixedModel, @formula(tsum ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject)), data);
    else
        throw(ArgumentError("Neither accuracy nor tsum in data."))
    end

end



# convert to R
rmodel = (model, data);
@rput rmodel


# export as file for import in R
R"saveRDS(rmodel, file = paste0('model_',$randint_str,'.rds'))" # the $ brings the julia variable into R

# R"save(model, file = 'model.rds')"


