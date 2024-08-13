# type1 error simulations

# to better understand the effect here: https://benediktehinger.de/blog/science/lmm-type-1-error-for-1condition1subject/
# but now for my data use case

#If you only have one trial per condition-level, there is no way to differentiate the random slope variability from the residual variability.
# In that case, leaving the random slope does not lead to higher type-1 errors. In other words, if you average within subject your conditions first, 
# then the random-intercept-only model (1|sub) is fine.

using DataFrames, StableRNGs, MixedModels, StatsBase, StatsPlots
using RCall, RData

# simulate a dataframe with 2 conditions, and 40 subjects, and y is the dependent variable, with one trial per subject per condition

iterations = 1000 # TODO increase to 1000

# Create an empty DataFrame with desired column names
df = DataFrame(modeltype = String31[], coef = String31[], rep = Int8[], subjects = Int32[], p = Float64[])

#nsub = 200 # repetitions per subject

for i in 1:iterations
    for nrep in [1, 10, 20]
        for nsub in [5,10,20,40,80, 120, 160, 200]
            # create a dataframe
            #rng = StableRNG(i)
            data = DataFrame(
                subject = repeat(1:nsub, inner=2*nrep),
                condition = repeat(["A", "B"], nsub*nrep),
                y = randn(2*nsub*nrep) # .+ 1
                #y = randn(rng, 2*nsub*nrep) # .+ 1
            )
            # give the intercept 
            #df.falsepos = ifelse.(df.p .<= 0.05, 1, 0)

            model_ri = fit(LinearMixedModel, @formula(y ~ 1 + condition + (1 | subject)), data)
            model_rs = fit(LinearMixedModel, @formula(y ~ 1 + condition + (1 + condition | subject)), data)

            # save the p values in vector
            table = DataFrame(coeftable(model_ri))
            ps = table[!,:"Pr(>|z|)"]
            row = (modeltype = "(1|sub)", coef = "global_intercept", rep = nrep, subjects = nsub, p = ps[1])
            push!(df, row)
            row = (modeltype = "(1|sub)", coef = "condition_effect", rep = nrep, subjects = nsub, p = ps[2])
            push!(df, row)

            table = DataFrame(coeftable(model_rs))
            ps = table[!,:"Pr(>|z|)"]
            row = (modeltype = "(1+cond|sub)", coef = "global_intercept", rep = nrep, subjects = nsub, p = ps[1])
            push!(df, row)
            row = (modeltype = "(1+cond|sub)", coef = "condition_effect", rep = nrep, subjects = nsub, p = ps[2])
            push!(df, row)
        end
    end        
end

# convert p to chose hypothesis
df.falsepos = ifelse.(df.p .<= 0.05, 1, 0)

result = combine(groupby(df, [:modeltype, :coef, :rep, :subjects]), :falsepos => mean)



#@rlibrary ggplot2
ggplot_object = R"""library(ggplot2); ggplot(data = $result, aes(x = subjects, y = falsepos_mean, group = rep, color = rep)) + geom_line() + facet_grid(modeltype ~ coef) + labs(title = "", x = "# of Subjects", y = "False Positive Rate", color="stimulus\nrepetitions") + geom_hline(aes(yintercept=0.05)) + scale_color_gradient(low = "darkred", high = "darkgreen") """


#
# next i simulate with 2 variables and an interaction
#
#
#

### model that looks like ours

df = DataFrame(modeltype = String31[], coef = String31[], rep = Int8[], subjects = Int32[], p = Float64[])

for i in 1:iterations
    # DEBUG
    #i=1

    nrep = 1
    nsub = 40
    npar = 4 # paraneters to extract from model
    nffx = 2 # 2 fixed effect params (before dummy code) without intercept
    levpffx = 2 # 2 levels per ffx

    # create a dataframe
    #rng = StableRNG(i)
    data = DataFrame(
        subject = repeat(1:nsub, inner=levpffx*nffx*nrep), # now we look at 4 different parameters
        alpha = repeat(["A", "B", "A", "B"], nsub*nrep),
        beta = repeat(["A", "B", "B", "A"], nsub*nrep),
        y = randn(nsub*levpffx*nffx*nrep) 
        #y = randn(rng, nsub*levpffx*nffx*nrep) 
    )
    # give the intercept 
    #df.falsepos = ifelse.(df.p .<= 0.05, 1, 0)

    model_ri = fit(LinearMixedModel, @formula(y ~ 1 + alpha + beta + (1 | subject)), data)
    model_rs = fit(LinearMixedModel, @formula(y ~ 1 + alpha + beta + (1 + alpha + beta | subject)), data)
    modeli_ri = fit(LinearMixedModel, @formula(y ~ 1 + alpha * beta + (1 | subject)), data)
    modeli_rs = fit(LinearMixedModel, @formula(y ~ 1 + alpha * beta + (1 + alpha * beta | subject)), data)

    # save the p values in vector
    table = DataFrame(coeftable(model_ri))
    ps = table[!,:"Pr(>|z|)"]
    row = (modeltype = "noint(1|sub)", coef = "global_intercept", rep = nrep, subjects = nsub, p = ps[1])
    push!(df, row)
    row = (modeltype = "noint(1|sub)", coef = "alpha_effect", rep = nrep, subjects = nsub, p = ps[2])
    push!(df, row)
    row = (modeltype = "noint(1|sub)", coef = "beta_effect", rep = nrep, subjects = nsub, p = ps[3])
    push!(df, row)
    row = (modeltype = "noint(1|sub)", coef = "int_effect", rep = nrep, subjects = nsub, p = NaN)
    push!(df, row)

    table = DataFrame(coeftable(model_rs))
    ps = table[!,:"Pr(>|z|)"]
    row = (modeltype = "noint(1+a+b|sub)", coef = "global_intercept", rep = nrep, subjects = nsub, p = ps[1])
    push!(df, row)
    row = (modeltype = "noint(1+a+b|sub)", coef = "alpha_effect", rep = nrep, subjects = nsub, p = ps[2])
    push!(df, row)
    row = (modeltype = "noint(1+a+b|sub)", coef = "beta_effect", rep = nrep, subjects = nsub, p = ps[3])
    push!(df, row)
    row = (modeltype = "noint(1+a+b|sub)", coef = "int_effect", rep = nrep, subjects = nsub, p = NaN)
    push!(df, row)

    table = DataFrame(coeftable(modeli_ri))
    ps = table[!,:"Pr(>|z|)"]
    row = (modeltype = "int+(1|sub)", coef = "global_intercept", rep = nrep, subjects = nsub, p = ps[1])
    push!(df, row)
    row = (modeltype = "int+(1|sub)", coef = "alpha_effect", rep = nrep, subjects = nsub, p = ps[2])
    push!(df, row)
    row = (modeltype = "int+(1|sub)", coef = "beta_effect", rep = nrep, subjects = nsub, p = ps[3])
    push!(df, row)
    row = (modeltype = "int+(1|sub)", coef = "int_effect", rep = nrep, subjects = nsub, p = ps[4])
    push!(df, row)

    table = DataFrame(coeftable(modeli_rs))
    ps = table[!,:"Pr(>|z|)"]
    row = (modeltype = "int+(1+a*b|sub)", coef = "global_intercept", rep = nrep, subjects = nsub, p = ps[1])
    push!(df, row)
    row = (modeltype = "int+(1+a*b|sub)", coef = "alpha_effect", rep = nrep, subjects = nsub, p = ps[2])
    push!(df, row)
    row = (modeltype = "int+(1+a*b|sub)", coef = "beta_effect", rep = nrep, subjects = nsub, p = ps[3])
    push!(df, row)
    row = (modeltype = "int+(1+a*b|sub)", coef = "int_effect", rep = nrep, subjects = nsub, p = ps[4])
    push!(df, row)


end

df.falsepos = ifelse.(df.p .<= 0.05, 1, 0)

result = combine(groupby(df, [:modeltype, :coef]), :falsepos => mean)

ggplot_object = R"""library(ggplot2); ggplot(data = $result, aes(x = modeltype, y = falsepos_mean, group = coef, fill = coef)) + geom_bar(stat = "identity", position="dodge") + labs(title = "", x = "model type", y = "False Positive Rate", color="coefficient") + geom_hline(aes(yintercept=0.05)) + scale_fill_discrete() """



# TODO shuffle my own data to check FPR

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




iterations = 1000

df = DataFrame(modeltype = String31[], coef = String31[], p = Float64[])
for i in 1:iterations
    for modeltype in ["(1|sub)", "(1+x|sub)"]

        # DEBUG
        #i=1
        #modeltype = "(1|sub))"
        # make a deepcopy of data
        idata = deepcopy(data)
        # shuffle accuracy column of idata
        idata[!, :accuracy] = shuffle(idata[!, :accuracy])

        if modeltype == "(1|sub)"
            formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + (1 | subject))
        elseif modeltype == "(1+x|sub)"
            formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + zerocorr( ref + hpf + lpf + emc + mac + base + det + ar | subject))
        end
        model = fit(LinearMixedModel, formula, idata) # suppress output into R

        # save the p values in vector
        table = DataFrame(coeftable(model))

        
        for row in eachrow(table)[2:end]
            row_new = (modeltype = modeltype, coef = row[1], p = row[5])
            push!(df, row_new)    
        end

    end
end

df.falsepos = ifelse.(df.p .<= 0.05, 1, 0)

result = combine(groupby(df, [:modeltype, :coef]), :falsepos => mean)

ggplot_object = R"""library(ggplot2); ggplot(data = $result, aes(x = coef, y = falsepos_mean, group = modeltype, fill = modeltype)) + geom_bar(stat = "identity", position="dodge") + labs(title = "Simulation of potential Type-I inflation by missing random slopes", x = "Term", y = "False Positive Rate", fill="Model Type") + geom_hline(aes(yintercept=0.05)) + scale_fill_discrete() """


