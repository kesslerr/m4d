
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
using CSV # to load own data
using RCall, RData # to control R or make it R objects

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

using Parsers # for getting input arguments
using DataFrames
using CSV
using Plots
using MixedModels

# https://github.com/palday/JellyMe4.jl
#ENV["LMER"] = "afex::lmer_alt" # set before import RCall and JellyMe4 to be able to convert zerocorr(rfx) correctly
using RCall
using JellyMe4

using RData
using CategoricalArrays # for categorical 


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
levels_ar = [false, true]; # was bool.... maybe check if works better with bool


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

# convert bool column to str
data.ar = [string(x) for x in data.ar]
data[!, :ar] = categorical(data[!, :ar], levels=["false", "true"]);



### show z/p values for FFX for models With and Without RFX slopes

formula_rs = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( ref + hpf + lpf + emc + mac + base + det + ar | subject));
model_rs = fit(LinearMixedModel, formula_rs, data);

formula_rsz = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + zerocorr( ref + hpf + lpf + emc + mac + base + det + ar | subject));
model_rsz = fit(LinearMixedModel, formula_rsz, data);

formula_ri = @formula(accuracy ~ ref + hpf + lpf + emc + mac + base + det + ar + ( 1 | subject));
model_ri = fit(LinearMixedModel, formula_ri, data);

#formula_rsi = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar)^2 + ( (1 + ref + hpf + lpf + emc + mac + base + det + ar)^2 | subject));
#model_rsi = fit(LinearMixedModel, formula_rsi, data);

formula_rszi = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 + zerocorr( (ref + hpf + lpf + emc + mac + base + det + ar) ^ 2 | subject));
model_rszi = fit(LinearMixedModel, formula_rszi, data);

formula_rii = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar)^2 + ( 1 | subject));
model_rii = fit(LinearMixedModel, formula_rii, data);


ffx_ri = coef(model_ri)
ffx_rs = coef(model_rs)
ffx_rsz = coef(model_rsz)
ffx_rii = coef(model_rii)[1:13]
#ffx_rsi = coef(model_rsi)[1:13]
ffx_rszi = coef(model_rszi)[1:13]

z_ri = model_ri.beta./ model_ri.stderror
z_rs = model_rs.beta./ model_rs.stderror
z_rsz = model_rsz.beta./ model_rsz.stderror
z_rii = model_rii.beta[1:13] ./ model_rii.stderror[1:13]
#z_rsi = model_rsi.beta./ model_rsi.stderror
z_rszi = model_rszi.beta[1:13] ./ model_rszi.stderror[1:13]

zs = vcat(z_ri, z_rs, z_rsz, z_rii, z_rszi) #z_rsi, 
ffx = vcat(ffx_ri, ffx_rs, ffx_rsz,ffx_rii,z_rszi) #z_rsi,
ffxnames = repeat(["a+b+(1|sub)", "a+b+(1+a+b|sub)", "a+b+(1+a+b||sub)","a*b+(1|sub)", "a*b+(1+a*b||sub)"], inner=13) #, "a*b+(1+a*b|sub)"
ffxind = repeat(1:13, outer=5)

df = DataFrame(
    model = ffxnames,
    effect = ffx,
    z = zs,
    predictor = ffxind
)



R"""
library(ggplot2)
library(dplyr)
df2 <- $df
df2 <- df2 %>% filter(predictor > 1)
ggplot(df2, aes(x=predictor, y=z, fill=model)) +
    geom_bar(stat = "identity", position="dodge") 
"""


# the sizes of the effects are identical !! but how about statistics?







using PlotlyJS
using Plotly

plot(scatter(x=1:10, y=rand(10), mode="markers"))



using Plots
x = range(0, 10, length=100)
y = sin.(x)
plot(x, y)

using CairoMakie


xs = range(0, 10, length = 30)
ys = 0.5 .* sin.(xs)

scatter(xs, ys)