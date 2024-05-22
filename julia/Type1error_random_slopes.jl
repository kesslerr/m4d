
# type I error tests when missing out random slopes
# fit model with and without random slopes and see how the z values of fixed effects parameters behave
# according to Bates et al you should rather fit the maximal model with all random slopes, else the type I error rate is inflated
# also described here: https://benediktehinger.de/blog/science/lmm-type-1-error-for-1condition1subject/


#using Parsers # for getting input arguments
using DataFrames
using CSV
using Plots
using MixedModels

# https://github.com/palday/JellyMe4.jl
ENV["LMER"] = "afex::lmer_alt" # set before import RCall and JellyMe4 to be able to convert zerocorr(rfx) correctly
# also, without lmer_alt env, the model makes some problems 
using RCall
using JellyMe4

using RData
using CategoricalArrays # for categorical 
using Parsers

using Suppressor # to supress the warnings and single steps in processing
# TODO: include the supressor https://github.com/JuliaIO/Suppressor.jl#usage

@suppress begin


    plot_file = ARGS[1]


    data_raw = CSV.read("../targets/eegnet_reordered.csv", DataFrame); #, types=column_types 
    #data_raw = filter(row -> row.dataset == "ERPCORE", data_raw);
    experiments = unique(data_raw.experiment)
    result_df = DataFrame()

    for expe in experiments
        println("Processing Experiment ", expe)
        data = filter(row -> row.experiment == expe, data_raw)
        select!(data, Not(:experiment)) #, :dataset

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

        formula_rs = @formula(accuracy ~ emc + mac + lpf + hpf + ref + base + det + ar + ( emc + mac + lpf + hpf + ref + base + det + ar | subject));
        model_rs = fit(LinearMixedModel, formula_rs, data);
        formula_rsz = @formula(accuracy ~ emc + mac + lpf + hpf + ref + base + det + ar + zerocorr( emc + mac + lpf + hpf + ref + base + det + ar | subject));
        model_rsz = fit(LinearMixedModel, formula_rsz, data);
        formula_ri = @formula(accuracy ~ emc + mac + lpf + hpf + ref + base + det + ar + ( 1 | subject));
        model_ri = fit(LinearMixedModel, formula_ri, data);
        #formula_rsi = @formula(accuracy ~ (ref + hpf + lpf + emc + mac + base + det + ar)^2 + ( (1 + ref + hpf + lpf + emc + mac + base + det + ar)^2 | subject));
        #model_rsi = fit(LinearMixedModel, formula_rsi, data);
        formula_rszi = @formula(accuracy ~ (emc + mac + lpf + hpf + ref + base + det + ar) ^ 2 + zerocorr( (emc + mac + lpf + hpf + ref + base + det + ar) ^ 2 | subject));
        model_rszi = fit(LinearMixedModel, formula_rszi, data);
        formula_rii = @formula(accuracy ~ (emc + mac + lpf + hpf + ref + base + det + ar)^2 + ( 1 | subject));
        model_rii = fit(LinearMixedModel, formula_rii, data);

        ffx_ri = coef(model_ri)
        ffx_rs = coef(model_rs)
        ffx_rsz = coef(model_rsz)
        ffx_rii = coef(model_rii)[1:13]
        #ffx_rsi = coef(model_rsi)[1:13]
        ffx_rszi = coef(model_rszi)[1:13]

        ffx_ri_names = coefnames(model_ri)
        ffx_rs_names = coefnames(model_rs)
        ffx_rsz_names = coefnames(model_rsz)
        ffx_rii_names = coefnames(model_rii)[1:13]
        #ffx_rsi_names = coefnames(model_rsi)[1:13]
        ffx_rszi_names = coefnames(model_rszi)[1:13]
        
        z_ri = model_ri.beta./ model_ri.stderror
        z_rs = model_rs.beta./ model_rs.stderror
        z_rsz = model_rsz.beta./ model_rsz.stderror
        z_rii = model_rii.beta[1:13] ./ model_rii.stderror[1:13]
        #z_rsi = model_rsi.beta./ model_rsi.stderror
        z_rszi = model_rszi.beta[1:13] ./ model_rszi.stderror[1:13]

        zs = vcat(z_ri, z_rs, z_rsz, z_rii, z_rszi) #z_rsi, 
        ffx = vcat(ffx_ri, ffx_rs, ffx_rsz,ffx_rii,z_rszi) #z_rsi,
        ffxparnames = vcat(ffx_ri_names, ffx_rs_names, ffx_rsz_names,ffx_rii_names,ffx_rszi_names) #z_rsi,
        ffxnames = repeat(["a+b+(1|sub)", "a+b+(1+a+b|sub)", "a+b+(1+a+b||sub)","a*b+(1|sub)", "a*b+(1+a*b||sub)"], inner=13) #, "a*b+(1+a*b|sub)"
        ffxnames = categorical(ffxnames, levels=["a+b+(1|sub)", "a+b+(1+a+b|sub)", "a+b+(1+a+b||sub)","a*b+(1|sub)", "a*b+(1+a*b||sub)"]);
        #ffxnames_factor = factor(ffxnames, levels=unique(ffxnames), ordered=true)
        ffxind = repeat(1:13, outer=5)
        exp_rep = repeat([expe], outer=13*5);

        df = DataFrame(
            model = ffxnames,
            effect = ffx,
            z = zs,
            parameter = ffxparnames,
            predictor = ffxind,
            experiment = exp_rep
        )
        append!(result_df, df)

    end

    R"""
    library(ggplot2)
    library(dplyr)
    df2 <- $result_df
    df2 <- df2 %>% filter(predictor > 1)
    ggplot(df2, aes(x=parameter, y=z, fill=model)) +
        geom_bar(stat = "identity", position="dodge") +
        facet_grid(experiment~., scales="free_y") +
        scale_fill_viridis_d(begin = 0., end = 0.9) + # skip the yellow
        #scale_y_continuous(trans='log2') + # if you want log scale on y
        labs(title="Fixed effect z values for different model types",
            x="Fixed effect parameter",
            y="z value",
            fill="Model type") +
        theme(axis.text.x = element_text(angle=90))
    ggsave($plot_file, dpi=300, height=20, width=15, units="cm")
    """

end