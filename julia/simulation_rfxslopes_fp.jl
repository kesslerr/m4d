

using DataFrames, StableRNGs, MixedModels, StatsBase, StatsPlots, CSV, Random
using CategoricalArrays # for categorical 
using GLM # the r2 function is in here
using Statistics # cor
using RCall, RData
using Parsers # for getting input arguments

plot_file = ARGS[1]


# to better understand the effect here: https://benediktehinger.de/blog/science/lmm-type-1-error-for-1condition1subject/
# but now for my data use case

data = CSV.read("../targets/eegnet.csv", DataFrame)
#only keep rows with experiment == "ERN"  (automatically drops MIPDB)
data = data[data.experiment .== "ERN", :]
# discard dataset column
select!(data, Not(:dataset))

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


iterations = 1000;

df = DataFrame(modeltype = String31[], coef = String31[], p = Float64[]);
for i in 1:iterations
    for modeltype in ["(1|sub)", "(1+x|sub)"]

        # DEBUG
        #i=1
        #modeltype = "(1|sub))"
        # make a deepcopy of data
        idata = deepcopy(data)
        # shuffle accuracy column of idata
        rng = StableRNG(i)
        idata[!, :accuracy] = shuffle(rng, idata[!, :accuracy])

        if modeltype == "(1|sub)"
            formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + (1 | subject));
        elseif modeltype == "(1+x|sub)"
            formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + zerocorr( ref + hpf + lpf + emc + mac + base + det + ar | subject));
        end
        model = fit(LinearMixedModel, formula, idata); # suppress output into R

        # save the p values in vector
        table = DataFrame(coeftable(model));

        
        for row in eachrow(table)[2:end]
            row_new = (modeltype = modeltype, coef = row[1], p = row[5]);
            push!(df, row_new);
        end

    end
end

df.falsepos = ifelse.(df.p .<= 0.05, 1, 0);

result = combine(groupby(df, [:modeltype, :coef]), :falsepos => mean);

#ggplot_object = 
R"""library(ggplot2); 
    ggplot(data = $result, aes(x = coef, y = falsepos_mean, group = modeltype, fill = modeltype)) + 
    geom_bar(stat = "identity", position="dodge") + 
    labs(title = "Simulation of potential Type-I inflation by missing random slopes", x = "Term", y = "False Positive Rate", fill="Model Type") + 
    geom_hline(aes(yintercept=0.05)) + 
    scale_fill_discrete();
    ggsave($plot_file, dpi=300, height=4, width=10) """


