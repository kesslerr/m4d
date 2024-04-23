### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ dbdc1896-9e79-11ed-024d-a387e81c3180
# import all Packages
begin
	using ColorSchemes
	using AlgebraOfGraphics#ggplot equivalent
	using CairoMakie #plotting 
	using MakieCore
	using MixedModels
	using Random
	using MixedModelsSim
	using PlutoUI #sliders
	using DataFrames
	using StatsBase#formulas
	using CategoricalArrays # categorical variables
	using DisplayAs # nicer LMM output
	using GLM # for LinearModel
	using HypothesisTests # for binomial errorbars
	using MakieCore # to automatically calculate binomErrors
end

# ╔═╡ 056f7f72-0cbb-4d3a-a726-7a99f488641a
module MakieDodge 
	using HTTP
	using MakieCore
	using Makie
	using CairoMakie
	 x = HTTP.download("https://gist.githubusercontent.com/behinger/8df41a3e051979a8e8ee0068f1aac6b8/raw/4c6f44603d4432b19c8abe2955eab4962a8ae45f/MakieDodge.jl",tempdir()) 
	include(x) 
	export(dodge)
	export(dodge!)
	export(Dodge)
end

# ╔═╡ 012201eb-adcf-47f6-8126-fd5582f2b32b

	md"""
	!!! explore
		change sliders below as you like, the type 1 error is automatically updated in the figure below
	"""
	

# ╔═╡ 556234b2-531e-424f-b790-e2fddff0e53e
begin
	md"""
| | |
|---|---|
|subjects |$(@bind nsub PlutoUI.Slider([2,4,8,16,32];default=16,show_value=true) )|
|trials per subject per level |$(@bind ntrials PlutoUI.Slider([1,2,4,8,16,32];default=8,show_value=true) )|

"""
end

# ╔═╡ c6f160b8-b17c-4d71-8508-c718d283148b
md"""
| | | |
|---|---|---|
|subj | intercept|$(@bind s_int  PlutoUI.Slider(0:0.5:2;default=1.5,show_value=true) )|
|subj | condition|$(@bind s_cond PlutoUI.Slider(0:0.5:2;default=1,show_value=true) )|
"""

# ╔═╡ 56ae8fde-6e74-40f9-8a31-7293729accc0
md"""
For the blogpost I used the values subjects=16, trials=8, intercept=1.5, condition=1.0
"""

# ╔═╡ 06f6f86e-1409-46b7-828d-02372f9ac794
md"""
# Code
What follows is just the code to generate everything :)
"""

# ╔═╡ b808036d-b9ad-4203-95ff-48a9dafdffd1
md"""
#### Initialize DataFrame and models
"""

# ╔═╡ 19955f36-127f-4a0b-8e7f-c9d9c8459a1b
begin
	# generate factorial
	d = factorproduct((;subject=nlevels(nsub, "S")),(;condition=nlevels(2,"C")),(;trials=1:ntrials))|>DataFrame
	d.y .= rand(size(d,1)) # dummy response

	form = @formula(y~1+condition+(1+condition|subject))
	m0 = fit(MixedModel, form, d)

	
	simulate!(m0,σ=1)
	d_sim = deepcopy(d)
	d_sim.y = m0.y
	d_avg = groupby(d_sim,[:subject,:condition])|>x->combine(x,:y=>mean)
	
	mFull = fit(MixedModel,@formula(y~1+condition+zerocorr(1+condition|subject)),d_sim)
	mInter = fit(MixedModel,@formula(y~1+condition+(1|subject)),d_sim)
	mSlope = fit(MixedModel,@formula(y~1+condition+(0+condition|subject)),d_sim)
	mAVG = fit(MixedModel,@formula(y_mean~1+condition+(1|subject)),d_avg)
	mLM = fit(LinearModel,@formula(y~1+condition),d_sim)


end;

# ╔═╡ a746540e-9a51-4659-8d61-b92b3d856b95
begin
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
			refit!(mSlope,m0.y)
		
			
			d_sim.y = m0.y
			
			mLM = fit(LinearModel,@formula(y~1+condition),d_sim)
			d_avg = groupby(d_sim,[:subject,:condition])|>x->combine(x,:y=>mean)
	
			refit!(mAVG,d_avg.y_mean)
			results[k] = (
			:m1=>extractP(mFull),
			:m2=>extractP(mInter),
			:m3=>extractP(mSlope),
			:m4=>extractP(mAVG),
			:m5=>extractP(mLM),
			:predictor=>coeftable(mFull)|>DataFrame|>x->x[:,1])
		end
		return vcat((DataFrame(r...) for r in results)...) |> x->stack(x,Not(:predictor),variable_name="model",value_name="pvalue")
	end

	
end;

# ╔═╡ b5162828-686a-4fe4-971b-aab492f0c479
md"""
# Type-1 Errors of various LMMs
We run $simulations simulations and analyse them with four models. We then check how many p-value¹ are <= α=0.05. We simulate no effect, thus we know the H₀ is actually always true. Consequently, our type-1 error should be 5%. It will not be perfectly 5% because we don't use many simulations and our test is not perfectly calibrated¹.


¹ We use $\frac{coef}{SE}$ which is mainly fast, not accurate, but the point holds for other procedures as well

---

"""

# ╔═╡ ef285287-1fd7-4b7e-986a-57f403e7ac48
md"""
#### Run Simulations $simulations times and extract P
"""

# ╔═╡ 45ce956d-fe5a-478b-ae67-d5444ee7ece2
md"""
#### Collect simulations & evaluate type-1
"""

# ╔═╡ df68fabe-cbd1-4727-b68d-44c14dd55f9b
function sim_type1(m0,d_sim;s_int=1.,s_cond=0,rep=simulations)
	
 	r = simModels(m0,d_sim,s_int,s_cond;rep=rep)|>
	x->groupby(x,[:predictor,:model])|>x->combine(x,:pvalue=>x->mean(x.<=0.05))#|>x->sort(x,:predictor)

	r.s_int .= s_int
	r.s_cond .= s_cond
	
	r.model = categorical(r.model)
	recode!(r.model, 
		"m1"=>"(1+c|sub)",
		"m2"=>"(1|sub)",
		"m3"=>"(0+c|sub)",
		"m4"=>"avg:(1|sub)",
		"m5"=>"no ranef",
	)
	

	return r
end;

# ╔═╡ 8e964d1f-9952-4c94-9d70-4b851b4d52e1
md"""
#### Plotting
"""

# ╔═╡ 4bf788ee-abd9-4d71-a8b5-6e5790499db1
begin
	# function to automatically calculate Binomial Errorbars
	import MakieCore.convert_arguments
	function MakieCore.convert_arguments(::Type{<:Errorbars}, x, y)
    xyerr = broadcast(x, y) do x, y
		e = 100. .*confint(BinomialTest(round(y.*(simulations./100. )),simulations),method=:wilson)
        Vec4f(x, y,y-e[1],e[2]-y )
    end
		
    (xyerr,)
end
end
	

# ╔═╡ cfc2d1da-bcfe-45ef-966c-814669f3b83e
begin
	
	dpl = data(sim_type1(m0,d_sim;s_int=s_int,s_cond=s_cond))
	sc = x->x.*100.
	dpl * mapping(
			:model=>"",
			:pvalue_function=>sc=>"type-1 error [%9",
			color=:predictor,dodge=:predictor)*	(
			visual(MakieDodge.Dodge,plot_fun=errorbars!,alpha=0.5)+
			visual(MakieDodge.Dodge,plot_fun=scatter!)
		)|>
	x->draw(x;axis=(
				title="Type-I from $simulations simulations with random effects: ($s_int + $s_cond|sub)\n (Errorbars depict wilsons binomial confidence intervals)",
				xticklabelrotation=π/8),
			palettes=(; color=ColorSchemes.Set1_3.colors))
	
	hlines!([5],color=:black,linestyle=:dot)
	ylims!(0,100)
	f = current_figure()
end;

# ╔═╡ 830843f2-4cec-4377-be8b-3798cf4b30de
begin
	f
end

# ╔═╡ b6927725-6be9-40b2-af61-e11af124624b
MakieDodge.Dogdge

# ╔═╡ e4d12cfe-f8d4-4225-b909-1704cc2003b7
md"""
#### Importing Packages
"""

# ╔═╡ 38c3bbcb-ab47-4fc0-9153-2f568300765c
md"""
we define a module here to get around Pluto.jl 's problem with including things :)
"""

# ╔═╡ Cell order:
# ╠═b5162828-686a-4fe4-971b-aab492f0c479
# ╟─012201eb-adcf-47f6-8126-fd5582f2b32b
# ╟─556234b2-531e-424f-b790-e2fddff0e53e
# ╟─c6f160b8-b17c-4d71-8508-c718d283148b
# ╟─830843f2-4cec-4377-be8b-3798cf4b30de
# ╟─56ae8fde-6e74-40f9-8a31-7293729accc0
# ╟─06f6f86e-1409-46b7-828d-02372f9ac794
# ╟─b808036d-b9ad-4203-95ff-48a9dafdffd1
# ╠═19955f36-127f-4a0b-8e7f-c9d9c8459a1b
# ╟─ef285287-1fd7-4b7e-986a-57f403e7ac48
# ╠═a746540e-9a51-4659-8d61-b92b3d856b95
# ╟─45ce956d-fe5a-478b-ae67-d5444ee7ece2
# ╠═df68fabe-cbd1-4727-b68d-44c14dd55f9b
# ╟─8e964d1f-9952-4c94-9d70-4b851b4d52e1
# ╠═4bf788ee-abd9-4d71-a8b5-6e5790499db1
# ╠═cfc2d1da-bcfe-45ef-966c-814669f3b83e
# ╠═b6927725-6be9-40b2-af61-e11af124624b
# ╟─e4d12cfe-f8d4-4225-b909-1704cc2003b7
# ╠═dbdc1896-9e79-11ed-024d-a387e81c3180
# ╟─38c3bbcb-ab47-4fc0-9153-2f568300765c
# ╠═056f7f72-0cbb-4d3a-a726-7a99f488641a
