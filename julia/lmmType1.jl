
# add to B Ehringer


using ColorSchemes
using AlgebraOfGraphics#ggplot equivalent
using CairoMakie #plotting 
using MakieCore
using MixedModels
using Random
using MixedModelsSim
#using PlutoUI #sliders
using DataFrames
using StatsBase#formulas
using CategoricalArrays # categorical variables
using DisplayAs # nicer LMM output
using GLM # for LinearModel
using HypothesisTests # for binomial errorbars
using MakieCore # to automatically calculate binomErrors
#using Plots
#using PyPlots
#pyplot()

#slider values
nsub = 40
ntrials = 1


# generate factorial
d = factorproduct((;subject=nlevels(nsub, "S")),(;condition=nlevels(2,"C")),(;trials=1:ntrials))|>DataFrame
d.y .= rand(size(d,1)) # dummy response

form = @formula(y~1+condition+(1+condition|subject))
m0 = fit(MixedModel, form, d)


simulate!(m0,σ=1) # !simulate replaces the y values in m0
d_sim = deepcopy(d)
d_sim.y = m0.y
d_avg = groupby(d_sim,[:subject,:condition])|>x->combine(x,:y=>mean)


mFull = fit(MixedModel,@formula(y~1+condition+zerocorr(1+condition|subject)),d_sim)

mInter = fit(MixedModel,@formula(y~1+condition+(1|subject)),d_sim)

#mSlope = fit(MixedModel,@formula(y~1+condition+(0+condition|subject)),d_sim)
#mAVG = fit(MixedModel,@formula(y_mean~1+condition+(1|subject)),d_avg)
mLM = fit(LinearModel,@formula(y~1+condition),d_sim)



# simulate

simulations=100;

extractP(x) = coeftable(x)|>DataFrame|>x->x[:,5];

function simModels(m0,d_sim,s_int,s_cond;rep=100)
    m0 = deepcopy(m0)
    update!(m0; subject=create_re(s_int,s_cond))
    d_sim = deepcopy(d_sim)
    results = Vector{Tuple}(undef,rep)
    for k = 1:rep
        simulate!(MersenneTwister(k),m0,σ=1,β=[0.,0])
        refit!(mFull,m0.y)
        refit!(mInter,m0.y)
        #refit!(mSlope,m0.y)
        d_sim.y = m0.y
        
        mLM = fit(LinearModel,@formula(y~1+condition),d_sim)
        #d_avg = groupby(d_sim,[:subject,:condition])|>x->combine(x,:y=>mean)
        #refit!(mAVG,d_avg.y_mean)

        results[k] = (
        :m1=>extractP(mFull),
        :m2=>extractP(mInter),
        #:m3=>extractP(mSlope),
        #:m4=>extractP(mAVG),
        :m3=>extractP(mLM),
        :predictor=>coeftable(mFull)|>DataFrame|>x->x[:,1])
    end
    return vcat((DataFrame(r...) for r in results)...) |> x->stack(x,Not(:predictor),variable_name="model",value_name="pvalue")
end


function sim_type1(m0,d_sim;s_int=1.,s_cond=0,rep=simulations)
   r = simModels(m0,d_sim,s_int,s_cond;rep=rep)|>
   x->groupby(x,[:predictor,:model])|>x->combine(x,:pvalue=>x->mean(x.<=0.05))#|>x->sort(x,:predictor)

   r.s_int .= s_int
   r.s_cond .= s_cond
   
   r.model = categorical(r.model)
   recode!(r.model, 
       "m1"=>"(1+c|sub)",
       "m2"=>"(1|sub)",
       #"m3"=>"(0+c|sub)",
       #"m4"=>"avg:(1|sub)",
       "m3"=>"no ranef",
   )
   return r
end;



# execute
sim_type1(m0,d_sim, s_int=1.,s_cond=1.0,rep=1000)



### MY OWN DATA STRUCTURE


#slider values
nsub = 40
ntrials = 1

# generate factorial
d = factorproduct((;subject=nlevels(nsub, "S")),
                  (;conditionA=nlevels(2,"A")),
                  (;conditionB=nlevels(2,"B")),
                  (;conditionC=nlevels(2,"C")),
                  (;conditionD=nlevels(2,"D")),
                  (;conditionE=nlevels(2,"E")),
                  (;trials=1:ntrials)
                  )|>DataFrame
d.y .= rand(size(d,1)) # dummy response

# he produces without zerocorr, but then uses zerocorr in estimation
form_rs_orig = @formula(y~1+conditionA+conditionB+conditionC+conditionD+conditionE+(1+conditionA+conditionB+conditionC+conditionD+conditionE|subject))
form_rs = @formula(y~1+conditionA+conditionB+conditionC+conditionD+conditionE+zerocorr(1+conditionA+conditionB+conditionC+conditionD+conditionE|subject))
form_ri = @formula(y~1+conditionA+conditionB+conditionC+conditionD+conditionE+(1|subject))
form_lm = @formula(y~1+conditionA+conditionB+conditionC+conditionD+conditionE)

m0 = fit(MixedModel, form_rs_orig, d)


simulate!(m0,σ=1) # !simulate replaces the y values in m0
d_sim = deepcopy(d)
d_sim.y = m0.y
#d_avg = groupby(d_sim,[:subject,:condition])|>x->combine(x,:y=>mean)

mFull = fit(MixedModel,form_rs,d_sim)
mInter = fit(MixedModel,form_ri,d_sim)
#mSlope = fit(MixedModel,@formula(y~1+condition+(0+condition|subject)),d_sim)
#mAVG = fit(MixedModel,@formula(y_mean~1+condition+(1|subject)),d_avg)
mLM = fit(LinearModel,form_lm,d_sim)



extractP(x) = coeftable(x)|>DataFrame|>x->x[:,5]; # TODO: are these only first 5 p values?

function simModels(m0,d_sim,s_int,s_condA,s_condB,s_condC,s_condD,s_condE;rep=100)
    m0 = deepcopy(m0)
    update!(m0; subject=create_re(s_int,s_condA,s_condB,s_condC,s_condD,s_condE))
    d_sim = deepcopy(d_sim)
    results = Vector{Tuple}(undef,rep)
    for k = 1:rep
        simulate!(MersenneTwister(k),m0,σ=1,β=[0.,0,0.,0,0.,0]) # was only 0.,0
        refit!(mFull,m0.y)
        refit!(mInter,m0.y)
        #refit!(mSlope,m0.y)
        d_sim.y = m0.y
        
        mLM = fit(LinearModel,form_lm,d_sim)
        #d_avg = groupby(d_sim,[:subject,:condition])|>x->combine(x,:y=>mean)
        #refit!(mAVG,d_avg.y_mean)

        results[k] = (
        :m1=>extractP(mFull),
        :m2=>extractP(mInter),
        #:m3=>extractP(mSlope),
        #:m4=>extractP(mAVG),
        :m3=>extractP(mLM),
        :predictor=>coeftable(mFull)|>DataFrame|>x->x[:,1])
    end
    return vcat((DataFrame(r...) for r in results)...) |> x->stack(x,Not(:predictor),variable_name="model",value_name="pvalue")
end


function sim_type1(m0,d_sim;s_int=1.,s_condA=0,s_condB=0,s_condC=0,s_condD=0,s_condE=0,rep=simulations)
   r = simModels(m0,d_sim,s_int,s_condA,s_condB,s_condC,s_condD,s_condE;rep=rep)|>
   x->groupby(x,[:predictor,:model])|>x->combine(x,:pvalue=>x->mean(x.<=0.05))#|>x->sort(x,:predictor)

   r.s_int .= s_int
   r.s_condA .= s_condA
   r.s_condB .= s_condB
   r.s_condC .= s_condC
   r.s_condD .= s_condD
   r.s_condE .= s_condE
   
   r.model = categorical(r.model)
   recode!(r.model, 
       "m1"=>"(1+c|sub)",
       "m2"=>"(1|sub)",
       #"m3"=>"(0+c|sub)",
       #"m4"=>"avg:(1|sub)",
       "m3"=>"no ranef",
   )
   return r
end;



# execute
results = sim_type1(m0,
        d_sim, 
        s_int=1.,
        s_condA=0.1, 
        s_condB=0.1,
        s_condC=0.,
        s_condD=0.,
        s_condE=0.,
        rep=1000)


using RCall, RData

#ggplot_object = 
R"""library(ggplot2); 

ggplot(data = $results, aes(x = model, y = pvalue_function, group = predictor, fill = predictor)) + 
geom_bar(stat = "identity", position="dodge") + 
labs(title = "Simulation of potential Type-I inflation by missing random slopes", x = "Model", y = "False Positive Rate", fill="Predictor") + 
geom_hline(aes(yintercept=0.05)) + 
scale_fill_grey(start=0.2, end=0.6)
#ggsave($plot_file, dpi=300, height=4, width=10) 
"""




### now fit a model with my own data, to estimate the actual effect sizes for modeling what happens when avoiding the RFX
using CSV
using DataFrames

data = CSV.read("../targets/eegnet.csv", DataFrame) #, types=column_types 
data = filter(row -> row.experiment == "ERN", data)
select!(data, Not(:experiment, :dataset))



# convert to factors
levels_hpf = ["None", "0.1", "0.5"];
levels_lpf = ["None", "6", "20", "45"];
levels_ref = ["average", "Cz", "P9P10"];
levels_emc = ["None", "ica"];
levels_mac = ["None", "ica"];
levels_base = ["200ms", "400ms"];
levels_det = ["offset", "linear"];
levels_ar = [false, true];


data[!, :hpf] = categorical(data[!, :hpf], levels=levels_hpf);
data[!, :lpf] = categorical(data[!, :lpf], levels=levels_lpf);
data[!, :ref] = categorical(data[!, :ref], levels=levels_ref);
data[!, :emc] = categorical(data[!, :emc], levels=levels_emc);
data[!, :mac] = categorical(data[!, :mac], levels=levels_mac);
data[!, :base] = categorical(data[!, :base], levels=levels_base);
data[!, :det] = categorical(data[!, :det], levels=levels_det);
data[!, :ar] = categorical(data[!, :ar], levels=levels_ar);

formula = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject))


model = fit(LinearMixedModel, formula, data) # suppress output into R


matrix = vcov(model)

R"""
library(ggplot2)
library(reshape2)
df <- melt($matrix*1000000)
print(df)
ggplot(df, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_continuous(type="viridis", trans = "log2") +
  labs(x="X-axis", y="Y-axis", title="Heatmap")
"""


R"""
library(ggplot2)
library(reshape2)
df <- melt($matrix)
print(df)
ggplot(df, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_continuous(type="viridis") +
  labs(x="X-axis", y="Y-axis", title="Heatmap")
"""