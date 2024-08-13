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

# define column type so that the strings (in experiment) is long enough
column_types = Dict(
    "experiment" => String31,  # Assuming you have a column named "experiment" in your CSV file
    # Add more column types if needed
)

# import dataframe
data_raw = CSV.read(string("../models/sliding_avgacc_single_extended.csv"), DataFrame, types=column_types);


unique_experiments = unique(data_raw.experiment)

for exp in unique_experiments
    println("Processing Experiment ", exp)
    data = deepcopy(data_raw)

    data = filter(row -> row.experiment == exp, data)
    select!(data, Not(:experiment))

    # convert to factors
    levels_hpf = ["None", "0.1", "0.5"];
    levels_lpf = ["None", "6", "20", "45"];
    levels_ref = ["average", "Cz", "P9P10"];
    levels_emc = ["None", "ica"];
    levels_mac = ["None", "ica"];
    levels_base = ["None", "200ms", "400ms"];
    levels_det = ["None", "linear"];
    levels_ar = ["False", "int", "intrej"];
    levels_experiment = ["ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3"];

    data[!, :hpf] = categorical(data[!, :hpf], levels=levels_hpf);
    data[!, :lpf] = categorical(data[!, :lpf], levels=levels_lpf);
    data[!, :ref] = categorical(data[!, :ref], levels=levels_ref);
    data[!, :emc] = categorical(data[!, :emc], levels=levels_emc);
    data[!, :mac] = categorical(data[!, :mac], levels=levels_mac);
    data[!, :base] = categorical(data[!, :base], levels=levels_base);
    data[!, :det] = categorical(data[!, :det], levels=levels_det);
    data[!, :ar] = categorical(data[!, :ar], levels=levels_ar);

    # model formula
    formula = @formula(accuracy ~ ( emc + mac + lpf + hpf + ref + det + base + ar) ^ 2 + zerocorr((emc + mac + lpf + hpf + ref + det + base + ar) ^ 2 | subject));

    # fit
    model = fit(LinearMixedModel, formula, data); # suppress output into R


    # convert to R
    rmodel = (model, data);
    #R"install.packages('afex', type = 'source')"
    #R"library('afex')" # needed for converting an lmer model with zerocorr

    elapsed_time = @elapsed begin
        #@rput rmodel
        @rput rmodel
    end



    # export as file for import in R
    R"saveRDS(rmodel, file = paste0('model_',$exp,'_tr.rds'))" # the $ brings the julia variable into R
    

end