library(JuliaCall)
options(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")
#julia_executable <- "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/julia"
julia_setup(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")

# colormap from the numerosity paper
colors_dark <- c("#851e3e", "#4f7871", "#3c1d85") # red, green, purple "#537d7d", 
colors_light <- c("#f6eaef", "#f2fefe", "#9682c0")


# rename variables
replacements <- list(
  "hpf" = "high pass", # [Hz]
  "lpf" = "low pass", # [Hz]
  "ref" = "reference",
  "ar" = "autoreject",
  "mac" = "muscle",
  "emc" = "ocular",
  "base" = "baseline",
  "det" = "detrending",
  "0.1" = "0.1 Hz",
  "0.5" = "0.5 Hz",
  "6" = "6 Hz",
  "20" = "20 Hz",
  "45" = "45 Hz",
  #"FALSE" = "False",
  #"false" = "False",
  "FALSE" = "None",
  "false" = "None",
  "TRUE" = "True",
  "true" = "True",
  "int" = "interpolate",
  "intrej" = "reject",
  "ica" = "ICA",
  "200ms" = "200 ms",
  "400ms" = "400 ms",
  "ica" = "ICA",
  "P9P10" = "P9/P10"
)
replacements_sparse <- list(
  "hpf" = "high pass", # [Hz]
  "lpf" = "low pass", # [Hz]
  "ref" = "reference",
  "ar" = "autoreject",
  "mac" = "muscle",
  "emc" = "ocular",
  "base" = "baseline",
  "det" = "detrending",
  #"0.1" = "0.1 Hz",
  #"0.5" = "0.5 Hz",
  #"6" = "6 Hz",
  #"20" = "20 Hz",
  #"45" = "45 Hz",
  "FALSE" = "None",
  "false" = "None",
  "TRUE" = "True",
  "true" = "True",
  "int" = "interp",
  "intrej" = "reject",
  "ica" = "ICA",
  "200ms" = "200",
  "400ms" = "400",
  "ica" = "ICA",
  "P9P10" = "P9/P10"
)

# own colorscale for factor levels
cols <- c("None" = "black",
          "0.1" = colors_dark[1],    
          "0.5" = colors_dark[2],     
          "6" = colors_dark[1],       
          "20" = colors_dark[2],     
          "45" = colors_dark[3],      
          "ICA" = colors_dark[1],     
          #"200ms" = "black",   
          "200" = colors_dark[1],   
          "400" = colors_dark[2],   
          #"offset" = "black",  
          "linear" = colors_dark[1], 
          #"false" = "black",
          "interp" = colors_dark[1],
          "reject" = colors_dark[2],
          "average" = "black",
          "Cz" = colors_dark[1],
          "P9/P10" = colors_dark[2]
)  

get_preprocess_data <- function(file) {
  data <- read_csv(file, col_types = cols())
  
  # change column order for arbitrary reason: None in LPF should be last, but because None is a factor both in hpf in lpf, lpf should come first with none as last entry, then later in the analyses when putting both in a long dataframe it will be ordered correctly
  #data <- data %>%
  #  select(names(data)[1:2], lpf, hpf, everything()) 
  # but keep None first here, because this will be the reference for HLM, and makes it therefore easier to interpret
  # recode lpf levels later
  # DOESNT work becuase the variables are ordered by name later and HPF will be BEFORE LPF...
  
  # new, preprocess already
  data$hpf <- factor(data$hpf, levels = c("None", "0.1", "0.5"))
  data$lpf <- factor(data$lpf, levels = c("None", "6", "20", "45"))
  data$ref <- factor(data$ref, levels = c("average", "Cz", "P9P10"))
  data$emc <- factor(data$emc, levels = c("None", "ica"))
  data$mac <- factor(data$mac, levels = c("None", "ica"))
  data$base <- factor(data$base, levels = c("None", "200ms", "400ms"))
  data$det <- factor(data$det, levels = c("None", "linear"))
  data$ar <- factor(tolower(data$ar), levels = c("false", "int", "intrej"))
  #data$ar <- factor(data$ar, levels = c("FALSE", "TRUE"))
  data$experiment <- factor(data$experiment, levels = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")) #, "LRP_6-9", "LRP_10-11", "LRP_12-13", "LRP_14-17", "LRP_18+", "6-9", "10-11", "12-13", "14-17", "18+"
  #data$dataset <- factor(data$dataset)
  
  if ("dataset" %in% names(data)) {
    data <- subset(data, select = -c(dataset))
  }
  
  # new: replace with paper-ready variable names / factor levels
  # col names
  #names(data) <- recode(names(data), !!!replacements)
  # NOT DONE, as this would disrupt short variable naming during modeling
  
  data
}

rjulia_r2 <- function(data){
  julia_library("Parsers, DataFrames, CSV, Plots, MixedModels, RData, CategoricalArrays, RCall, JellyMe4, GLM, Statistics")
  df <- data.frame(model=c(),
                   experiment=c(), 
                   interactions=c(), 
                   metric=c(),
                   value=c())
  for (interaction in c( "true", "false")){
    for (thisExperiment in unique(data$experiment)){
      for (modeltype in c("EEGNet", "Time-resolved")){
        data_tmp <- data %>% filter(experiment == thisExperiment)
        if (modeltype == "EEGNet"){
          data_tmp <- data_tmp %>% filter(!is.na(accuracy))
        } else if (modeltype == "Time-resolved"){
          data_tmp <- data_tmp %>% filter(!is.na(tsum))
        }
        
        if (modeltype == "EEGNet"){
          julia_assign("data", data_tmp) # bring data into julia
          if (interaction == "true"){
            julia_command("formula = @formula(accuracy ~ (emc + mac + lpf + hpf + ref + det + base + ar) ^ 2 + zerocorr( (emc + mac + lpf + hpf + ref + det + base + ar) ^ 2 | subject));")
          } else if (interaction == "false") {
            julia_command("formula = @formula(accuracy ~ emc + mac + lpf + hpf + ref + det + base + ar + ( emc + mac + lpf + hpf + ref + det + base + ar | subject));")
          }
          julia_command("model = fit(LinearMixedModel, formula, data);")
          julia_command("predictions = predict(model, data);")
          ir2 <- julia_eval("cor(predictions, data.accuracy)^2;")
          iaic <- julia_eval("aic(model);")
          iloglikelihood <- julia_eval("loglikelihood(model);")
          
        } else if (modeltype == "Time-resolved"){
            if (interaction=="true"){
              mod <- lm(formula="tsum ~ (emc + mac + lpf + hpf + ref + det + base + ar) ^ 2",
                 data = data_tmp)
            } else if (interaction=="false"){
              mod <- lm(formula="tsum ~ emc + mac + lpf + hpf + ref + det + base + ar",
                 data = data_tmp)
            }
          ir2 <- summary(mod)$r.squared
          iaic <- AIC(mod)
          iloglikelihood <- logLik(mod)[1]
        }
          
        df <- rbind(df, data.frame(model=modeltype,
                                   experiment=thisExperiment, 
                                   interactions=interaction, 
                                   metric="R2",
                                   value=ir2))
        df <- rbind(df, data.frame(model=modeltype,
                                   experiment=thisExperiment, 
                                   interactions=interaction, 
                                   metric="AIC",
                                   value=iaic))
        df <- rbind(df, data.frame(model=modeltype,
                                   experiment=thisExperiment, 
                                   interactions=interaction, 
                                   metric="Log Likelihood",
                                   value=iloglikelihood))
      }
    }
  }
  df
}

chord_plot <- function(plot_filepath){
  varnames <- c("emc","mac","lpf","hpf","ref","det","base","ar") #c("ref","hpf","lpf","emc","mac","det","base","ar")
  varnames <- recode(varnames, !!!replacements)
  numbers <- c(0,1,1,0,0,0,0,1,
               1,0,1,1,1,1,1,1,
               1,1,0,1,1,1,1,1,
               0,1,1,0,1,0,0,1,
               0,1,1,1,0,0,0,1,
               0,1,1,0,0,0,1,0,
               0,1,1,0,0,1,0,0,
               1,1,1,1,1,0,0,0)
  data <- matrix( numbers, ncol=8)
  rownames(data) <- varnames
  colnames(data) <- varnames
  col_fun = colorRamp2(range(data), c("#ddd8d5", "#4e4b44"), transparency = 0.5)
  png(plot_filepath, width=8, height=8, units="cm", res=300) 
  chordDiagram(data, 
               transparency = 0,
               symmetric = TRUE,
               big.gap = 20,
               small.gap = 5,
               link.visible = data > 0.5,
               grid.col = "black",
               col = col_fun,
               annotationTrack =  c("name", "grid") # remove xticks / xticklabels
               )
  dev.off() # the chordDiagram is not saved into an object that can be returned, therefore save it to file
  plot_filepath
}



# sliding plots
sliding_plot_all <- function(data){
  data <- data %>%
    mutate(variable = recode(variable, !!!replacements))
  ggplot(data, aes(x=level, y=tsum)) +
    geom_bar(stat="identity") + 
    facet_wrap(. ~variable, scales="free_x")
}
sliding_plot_experiment <- function(data){
  data <- data %>%
    mutate(variable = recode(variable, !!!replacements))
  ggplot(data, aes(x=level, y=tsum)) +
    geom_bar(stat="identity", position=position_dodge()) + 
    facet_grid(experiment ~variable, scales = "free_x")
}

luckfps <- data.frame(
  experiment = c('ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3'),
  emc = c('ica', 'ica', 'ica', 'ica', 'ica', 'ica', 'ica'),
  mac = c('ica', 'ica', 'ica', 'ica', 'ica', 'ica', 'ica'),
  lpf = c('None', 'None', 'None', 'None', 'None', 'None', 'None'),
  hpf = c('0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1'),
  ref = c('P9P10', 'P9P10', 'P9P10', 'average', 'P9P10', 'P9P10', 'P9P10'),
  base = c('200ms', '200ms', '200ms', '200ms', '200ms', '200ms', '200ms'),
  det = c('None', 'None', 'None', 'None', 'None', 'None', 'None'),
  #ar = c('TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE')
  ar = c('int', 'int', 'int', 'int', 'int', 'int', 'int')
)

timeresolved_plot <- function(data){
  data_fp <- semi_join(data, luckfps, 
                       by = c("experiment", "emc", "mac", "lpf", "hpf", "ref", "det", "base", "ar")) #c("experiment", "ref", "hpf", "lpf", "emc", "mac", "det", "base", "ar")
  # TR-Decoding with points as significance markers
  ggplot(data_fp, aes(x = times, y = `balanced accuracy`)) +
    geom_line() +
    geom_hline(yintercept=0.5, linetype="solid") +
    geom_vline(xintercept=0, linetype="dashed") +
    geom_point(data=filter(data_fp, significance=="TRUE"),
               aes(x=times, y=0.48),
               color=colors_dark[3],
               size=1
    ) +

    facet_wrap(experiment~., scales = "free_x", ncol=1) +
    scale_x_continuous(breaks = seq(-8, 8, by = 2)/10, 
                       labels = seq(-8, 8, by = 2)/10) +
    labs(x="Time [s]", y="Accuracy") +
         #title="Time-Resolved Decoding Results - Exemplary Single Forking Path")
    #theme_classic()
    theme_grey()
  
}



raincloud_mm <- function(data, title = ""){
  # Apply replacements batchwise across all columns
  data <- data %>%
    mutate(variable = recode(variable, !!!replacements))
  
  # https://rpubs.com/rana2hin/raincloud
  ggplot(data, aes(x = factor, y = accuracy)) +
    
    # add half-violin from {ggdist} package
    stat_halfeye(
      # adjust bandwidth
      adjust = 0.5,
      # move to the right
      justification = -0.2,
      # remove the slub interval
      .width = 0,
      point_colour = NA,
      scale = 0.5 ##  <(**new parameter**)
    ) +
    geom_boxplot(
      width = 0.12,
      # removing outliers
      outlier.color = NA,
      alpha = 0.5
    ) +
    stat_dots(
      # ploting on left side
      side = "left",
      # adjusting position
      justification = 1.1,
      # adjust grouping (binning) of observations
      binwidth = 0.005,
      
    ) +
    facet_wrap(~variable, scales="free") +
    labs(title = title,
         x="preprocessing step",
         y="accuracy")

}

# raw accuracies in a raincloud plot
raincloud_acc <- function(data, title = ""){
  names(data)[1] <- str_to_title(names(data)[1])
  DV <- names(data)[1]
  
  # https://rpubs.com/rana2hin/raincloud
  p <- ggplot(data, aes(x = experiment, y = !!sym(DV) )) +
    # DEPRECATED: aes_string(x = "Experiment", y = DV))
    # add half-violin from {ggdist} package
    stat_halfeye(
      # adjust bandwidth
      adjust = 0.5,
      # move to the right
      justification = -0.1,
      # remove the slub interval
      .width = 0,
      point_colour = NA,
      scale = 0.8 ##  <(**new parameter**)
    ) +
    geom_boxplot(
      width = 0.12,
      # removing outliers
      outlier.color = NA,
      alpha = 0.5
    ) +
    labs(title = title,
         x="Experiment",
         y=DV)
  if (DV=="Accuracy"){
    p <- p + geom_hline(yintercept=0.5, lty="dashed")
    p <- p + lims(y=c(0.46,NA))
  }
  p
}


paired_box <- function(data, title=""){
  # Apply replacements batchwise across all columns
  data <- data %>%
    mutate(variable = recode(variable, !!!replacements))
  
  # boxplot with paired point plot
  ggpaired(filter(data, variable != "experiment"), # exclude the experiment variable if present
           x = "factor", 
           y = "accuracy",
           id = "subject",
           line.color = "subject", #"gray",
           title = title,
           line.size = 0.6
           #facet.by = c("variable") # this one does not work for different factors per facet
  ) + 
    facet_grid(.~variable, scales="free") +
    theme(legend.position = "none") +
    labs(x="preprocessing step",
         y="accuracy")
}



## HLM and EMM

est_emm <- function(model, orig_data){
  # DEBUG
  #model = return_model
  #data <- tar_read(data_tsum_exp, branches=1)
  #data <- data %>% filter(experiment=="ERN")
  #model <- tar_read(sliding_LMi2, branches=1)
  #model <- model[[1]]
  
  
  variables = c("emc","mac","lpf","hpf","ref","det","base","ar") #c("ref", "hpf","lpf","emc","mac","det","base","ar")
  experiment = unique(orig_data$experiment)
  means = data.frame()
  contra = data.frame()
  fs = data.frame()
  for (variable in variables){
    # MAIN EFFECTS (1 factor)
    emm <- emmeans(model, 
                   specs = formula(paste0(c("pairwise ~ ",variable))), 
                   lmer.df = "asymp", # to supress warning: Note: D.f. calculations have been disabled because the number of observations
                   #lmerTest.limit = 322560,
                   #pbkrtest.limit = 322560) # to not have inf df
                   data=orig_data,# new, data not found in model (also works for HLM?)
    )
    
    # get means
    dfw <- emm$emmeans %>% 
      as.data.frame() # leaving out contrasts for now
    dfw$variable <- names(dfw)[1]
    names(dfw)[1] <- "level"
    dfw <- dfw[, c(7, 1, 2)]  # CAVE: the SD/CIs can not be used (see warning and values), therefore cutting them
    if (class(dfw$level) == "logical"){ # to avoid TRUE and FALSE being converted to NA (in variable="ar")
      dfw$level <- as.factor(dfw$level)
    }
    #dfw$level <- as.character(dfw$level) # to avoid TRUE and FALSE being converted to NA (in variable="ar")
    means <- rbind(means, dfw)
    
    # get contrasts
    dfc <- emm$contrasts %>% 
      as.data.frame() %>% # leaving out contrasts for now
      mutate(variable = variable) %>%
      separate(contrast, c("level.1", "level.2"), " - ")
    dfc <- dfc[, c(8, 1, 2, 3, 4, 5, 6, 7)]
    contra <- rbind(contra, dfc)
    
  }
  # omnibus tests for each factor
  fs <- joint_tests(model, data=orig_data)
  
  # significance asterisks
  contra <- contra %>% mutate(significance = stars.pval(.$p.value) )
  fs %<>% mutate(p.fdr = p.adjust(.$p.value, "BY", length(.$p.value))) %>%
    mutate(sign.unc = stars.pval(.$p.value)) %>%
    mutate(sign.fdr = stars.pval(.$p.fdr))
  
  # add experiment variable to all
  means %<>% mutate(experiment = experiment) %>% select(experiment, everything())
  contra %<>% mutate(experiment = experiment) %>% select(experiment, everything())
  fs %<>% mutate(experiment = experiment) %>% select(experiment, everything())
  
  return(list(means, contra, fs))
  
}

est_emm_int <- function(model, data){
  experiment <- experiment <- unique(data$experiment)
  variables = c("emc","mac","lpf","hpf","ref","det","base","ar") 
  means = data.frame()
  contra = data.frame()
  for (variable.1 in variables) {
    for (variable.2 in variables) {
      if (variable.1 != variable.2) {
        #print(paste(variable.1, variable.2))
        
        # extract marginal means grouped for results and stats
        emm <- emmeans(model, 
                       as.formula(paste("pairwise ~", variable.1, "|", variable.2)), 
                       data=data,
                       lmer.df = "asymp" # to supress warning: Note: D.f. calculations have been disabled because the number of observations
        )
        
        # means
        dfw <- emm$emmeans %>% 
          as.data.frame() # leaving out contrasts for now
        dfw$variable.1 <- names(dfw)[1] # grouping variable 
        dfw$variable.2 <- names(dfw)[2] 
        names(dfw)[1] <- "level.1" # grouping variable
        names(dfw)[2] <- "level.2" 
        dfw <- dfw[, c(8, 9, 1, 2, 3)]  # CAVE: the SD/CIs can not be used (see warning and values), therefore cutting them
        # to avoid TRUE and FALSE being converted to NA (in variable="ar")
        if (class(dfw$level.1) == "logical") {dfw$level <- as.factor(dfw$level.1)}
        if (class(dfw$level.2) == "logical") {dfw$level <- as.factor(dfw$level.2)}
        means <- rbind(means, dfw)
        
        # contrasts
        dfc <- emm$contrasts %>% 
          as.data.frame() %>% # leaving out contrasts for now
          mutate(variable.1 = variable.1) %>%
          mutate(variable.2 = variable.2) %>%
          separate(contrast, c("level.1.1", "level.1.2"), " - ")
        names(dfc)[3] <- "level.2"
        dfc <- dfc[, c(9, 10, 1, 2, 3, 4, 5, 6, 7, 8)]
        contra <- rbind(contra, dfc)
        
        # f test # TODO: check if and how to do it
        #f <- joint_tests(emm)
        
        # extract marginal means grouped as plotting information
        #tmp <- emmip(model, as.formula(paste(variable.1, "~", variable.2)), data=data, plotit=FALSE)
        #tmp$variable.1 <- names(tmp)[1]
        #tmp$variable.2 <- names(tmp)[2]
        #names(tmp)[1:2] <- c("level1","level2")
        #results <- rbind(results, tmp)
        #emmip(model, ref ~ hpf, data=data)
      }
    }
  }
  # significance asterisks
  contra <- contra %>% mutate(significance = stars.pval(.$p.value) )
  #fs %<>% mutate(p.fdr = p.adjust(.$p.value, "BY", length(.$p.value))) %>% # TODO: write in manuscript that now BY correction is done per experiment!!
  #  mutate(sign.unc = stars.pval(.$p.value)) %>%
  #  mutate(sign.fdr = stars.pval(.$p.fdr))
  
  # add experiment variable to all
  means %<>% mutate(experiment = experiment) %>% select(experiment, everything())
  contra %<>% mutate(experiment = experiment) %>% select(experiment, everything())
  #fs %<>% mutate(experiment = experiment) %>% select(experiment, everything())
  
  return(list(means, contra))
}


# heatmap of emms
heatmap <- function(data){
  data <- data %>% 
    reorder_variables(column_name = "variable") %>%
    relevel_variables(column_name = "level") %>%
    # Apply replacements batchwise across all columns
    mutate(variable = recode(variable, !!!replacements)) %>%
    # NEW: replacements for some levels, to not overload the image too much
    mutate(level = recode(level, !!!replacements_sparse)) %>%
    # delete the experiment compairson in the full data
    #filter(!(experiment == "ALL" & variable == "experiment")) %>% 
    # center around zero for better comparability
    group_by(experiment) %>%
    mutate(emmean = (emmean / mean(emmean) - 1) * 100 ) # now it is percent

  ggplot(data, aes(y = 0, x = level, fill = emmean)) +
    geom_tile() +
    facet_grid(experiment~variable, scales="free") +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          strip.text.y = element_text(angle=0)) + # rotate the experiment labels
    #scale_fill_continuous_diverging(palette = "Blue-Red 3", 
    #                                l1 = 45, # luminance at endpoints
    #                                l2 = 100, # luminance at midpoints
    #                                p1 = .9, 
    #                                p2 = 1.2) +
    #scale_fill_gradientn(colours=brewer.prgn(100), guide = "colourbar") +
    #scale_fill_gradientn(colours = c(colors_dark[1], "white", colors_dark[2]), # numerosity colors
    #                     #values = scales::rescale(c(-2, -0.5, 0, 0.5, 2))
    #                     ) +
    #scale_fill_gradient2(cetcolor::cet_pal(5, "d3")) +  
    #scale_fill_gradient2(low=cetcolor::cet_pal(2, "d3")[1], mid="white", high=cetcolor::cet_pal(2, "d3")[2]) + 
    scale_fill_gradient2(low=colors_dark[1], mid="white", high=colors_dark[2]) + 
    labs(x="processing step",
         y="",
         #fill="% change\naccuracy")  
         fill="% deviation from\nmarginal mean")  
        # Percentage marginal mean discrepancy
        # Distance from average (in %)
        # Percent above/below average
  
}


reorder_variables <- function(data, column_name){
  # reorder the factor levels of the variables in the following order
  #new_order = c("ref", "hpf","lpf","emc","mac","det","base","ar") # original
  new_order = c("emc","mac","lpf","hpf","ref","det","base","ar") #c("ref", "lpf","hpf","emc","mac","det","base","ar") # I CHANGED HPF AND LPF
  data[[column_name]] <- factor(data[[column_name]], levels = new_order)  
  return(data)
}

relevel_variables <- function(data, column_name){
  # reorder the factor levels of the variables in the following order
  new_order = c("average", "Cz", "P9P10", "6", "20", "45","None","0.1", "0.5","ica", "200ms", "400ms", "linear", "false", "int", "intrej") # TODO double check if it is correct with the new MV3
  data[[column_name]] <- factor(data[[column_name]], levels = new_order)  
  return(data)
}

interaction_plot <- function(means, title_prefix=""){
  # DEBUG
  #means = tar_read(eegnet_HLMi2_emm_means, branches=1)
  #means <- means %>% filter(experiment=="ERN")
  
  experiment <- unique(means$experiment)
  means %<>% 
    reorder_variables("variable.1") %>%
    reorder_variables("variable.2") %>%
    relevel_variables("level.1") %>%
    relevel_variables("level.2") 
  #means <- 
  meansr <- means %>% 
    mutate(variable.1 = recode(variable.1, !!!replacements_sparse)) %>%
    mutate(variable.2 = recode(variable.2, !!!replacements_sparse)) %>%
    mutate(level.1 = recode(level.1, !!!replacements_sparse)) %>%
    mutate(level.2 = recode(level.2, !!!replacements_sparse))  

  p1 <- ggplot(meansr, 
               aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
    geom_line(size = 1.2) + 
    facet_grid(variable.2~variable.1, scales = "free") +
    labs(title = paste0(title_prefix,experiment),
         y = "Marginal Mean", 
         x = "processing step", 
         color = "Group: ") + # legend removed anway
    scale_color_manual(values=cols) +
    theme_classic() +
    scale_x_discrete(expand = c(0.2, 0.0)) + # strech a bit in x direction
    theme(legend.position = "none")  # Remove legend
  
  
  # make pseudo plots for each row, and extract only the legend
  variable.2s <- sort(unique(meansr$variable.2))
  legends <- list()
  for (v2 in variable.2s){
    results_filtered <- meansr %>% filter(variable.2 == v2)
    ptmp <- ggplot(results_filtered, 
                   aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
      geom_line(size = 1.2) + 
      facet_grid(.~variable.1, scales = "free") +
      labs(color = paste0("Group: ",v2)) +
      scale_color_manual(values=cols) +
      theme_classic()
    # get legend
    legend <- as_ggplot(ggpubr::get_legend(ptmp))
    legends <- c(legends, list(legend))
  }
  
  # possibility 1: legend at the right side
  # p1_and_legends <- c(list(p1), legends)
  # grid.arrange(grobs=p1_and_legends, #, ncol = 5)y
  #           layout_matrix = matrix(c(1,1,1,1,1,1,1,2,
  #                                    1,1,1,1,1,1,1,3,
  #                                    1,1,1,1,1,1,1,4,
  #                                    1,1,1,1,1,1,1,5,
  #                                    1,1,1,1,1,1,1,6,
  #                                    1,1,1,1,1,1,1,7,
  #                                    1,1,1,1,1,1,1,8,
  #                                    1,1,1,1,1,1,1,9),
  #                                  nrow=8, byrow=TRUE))
  
  # possibility 2: on diagonals
  cow <- cowplot::ggdraw() + 
    cowplot::draw_plot(p1, x = 0, y = 0, width = 1.0, height = 1.0) 
  
  d <- 1/8.5
  e <- 1/9
  for (i in 0:7){
    # single legends
    cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                    x = d*i+0.05, 
                                    y = 0.9-d*i, 
                                    width = 0.1, height = 0.03)
    # horizontal lines between facets
    if (i<7){
      cow <- cow + cowplot::draw_line(x = c(0.05, 0.97), 
                                      y = c(0.85-d*i, 0.85-d*i), 
                                      color = "grey", size = 0.5)
    }
  }
  cow
}

qqplot <- function (model, data="") # argument: vector of numbers
{
  if (is.data.frame(data)){
    title=unique(data$experiment)
  }
  if (length(title) > 1){
    title="ALL"
  }
  vec <- resid(model)
  # following four lines from base R's qqline()
  y <- quantile(vec[!is.na(vec)], c(0.25, 0.75))
  x <- qnorm(c(0.25, 0.75))
  slope <- diff(y)/diff(x)
  int <- y[1L] - slope * x[1L]
  d <- data.frame(resids = vec)
  ggplot(d, aes(sample = resids)) + 
    stat_qq() + 
    geom_abline(slope = slope, intercept = int) +
    labs(title=title)
  
}

rvfplot <- function (model, data="") # argument: vector of numbers
{
  if (is.data.frame(data)){
    title=unique(data$experiment)
  }
  if (length(title) > 1){
    title="ALL"
  }
  amodel <- augment(model)
  if (dim(amodel)[1] > 1000){
    amodel <- sample_n(amodel, 1000) # NEW: reduce number of datapoints for computational reasons
  }
  ggplot(data = amodel, aes(x = .fitted, y = .resid)) +
    geom_point() +
    geom_smooth(method = "loess", se = FALSE) +
    labs(x = "Fitted Values", y = "Residuals", title = title)
}

sasrvfplot <- function (model, data="") # argument: vector of numbers
{
  if (is.data.frame(data)){
    title=unique(data$experiment)
  }
  if (length(title) > 1){
    title="ALL"
  }
  amodel <- augment(model)
  if (dim(amodel)[1] > 1000){
    amodel <- sample_n(amodel, 1000) # NEW: reduce number of datapoints for computational reasons
  }
  ggplot(data = amodel, aes(x = .fitted, y = sqrt(abs(.resid)))) +
    geom_point() +
    geom_smooth(method = "loess", se = FALSE) +
    labs(x = "Fitted Values", y = "sqrt ( abs ( Standardized Residuals ) )", title=title)
}


# RFX vis
rfx_vis <- function(model, orig_data){
  data <- ranef(model)$subject
  
  data_long <- data %>%
    pivot_longer(
      cols = names(.), #-c("subject"), #, # Select columns starting with "est"
      names_to = "level",         # Create the "level" column
      values_to = "mean" # Create the "conditional mean" column
    ) 
  if (any(startsWith(data_long$level, "experiment"))) {
    title <- "ALL" 
  } else {
    title <- unique(orig_data$experiment)
  } # TODO: can I get the experiment information from somewhere in the model?
  ggplot(data_long,
         aes(y=mean, x=level)) +
    geom_boxplot() +
    labs(y="Conditional Mean", x="Random Effects Term", title=title) +
    theme(axis.text.x = element_text(angle=90))
  
}

# RFX and sociodemographics
plot_rfx_demographics <- function(model, demographics, orig_data){
  # DEBUG
  #model <- tar_read(eegnet_HLMi2, branches=1)[[1]]
  #demographics <- tar_read(demographics)
  #orig_data <- tar_read(data_eegnet_exp) %>% filter(experiment == "ERN")
  
  # from rfx_vis function
  data <- lme4::ranef(model)$subject %>%
    mutate(Subject = rownames(.)) %>%
    mutate(Intercept = `(Intercept)`) %>%
    select(c(Intercept, Subject))
  rownames(data) <- NULL
  experiment <- unique(orig_data$experiment)
  
  # merge with demographics
  data <- left_join(data, demographics, c("Subject" = "participant_id"))
  
  # plot age
  p1 <- ggplot(data, aes(x=age, y=Intercept, color=sex)) +
    geom_point() +
    #geom_smooth(method="lm", se=TRUE) +
    geom_hline(aes(yintercept=0), lty="dashed") +
    labs(x="Age", y="Intercept") +
    scale_color_manual(values = colors_dark)
  
  p2 <- ggplot(data, aes(x=sex, y=Intercept, fill=sex)) +
    geom_boxplot(notch=FALSE) +
    geom_hline(aes(yintercept=0), lty="dashed") +
    labs(x="Sex", y="Intercept") +
    guides(fill = "none") +# remove legend for "fill"
    scale_fill_manual(values = colors_dark)
  
  p3 <- ggplot(data, aes(x=Intercept, fill=handedness)) +
    geom_histogram(bins=20) + 
    geom_vline(aes(xintercept=0), lty="dashed") +
    labs(x="Intercept", y="Participant Count") +
    #scale_fill_viridis_d(begin=0, end=0.8) +
    scale_fill_manual(values = c(colors_light[3], "grey"))
  
  #If it is not the first (ERN) plot, then remove all legends
  if (experiment == "ERN"){
    
    #p1 <- p1 + theme(legend.position = c(0.1, 0)) # position within figure bottom left
    #p3 <- p3 + theme(legend.position = c(0.1, 1)) # postition within figure top left
  } else {
    p3 <- p3 + theme(legend.position="none")
  } 
  p1 <- p1 + theme(legend.position="none")
  
  # Of ot os not the last (P3) plot, remove all X labels
  if (experiment != "P3"){
    p1 <- p1 + labs(x="")
    p2 <- p2 + labs(x="")
    p3 <- p3 + labs(x="")
  }
  
  # statistics on the results
  # 1. 2-sample test on male vs female with bonferroni correction (7 experiments)
  males <- data[data$sex == "Male", ]
  females <- data[data$sex == "Female", ]
  t_result <- t.test(males$Intercept, females$Intercept, var.equal = TRUE)
  adj_p <- p.adjust(t_result$p.value, method="BH", n=7) #bonferroni
  p2 <- p2 + annotate("text", x=1.5, y=0.15, label=paste("p=", sprintf("%.2f", adj_p)), color="black", size=4)
  
  # 2. are lines in p1 stat diff
  #agemodel <- lm(Intercept ~ age * sex, data = data)
  #if used, then interpret the interaction term but maybe also the slope
  
  ggarrange(p1,p2,p3, ncol=3) %>%
  annotate_figure(left = text_grob(experiment, 
                                    color = "black", face = "bold", size = 12, rot=90))
}

# extract ranef from all experiment HLM models
extract_rfx_exp <- function(model, orig_data){
  data <- ranef(model)$subject %>%
    mutate(Subject = rownames(.)) %>%
    mutate(Intercept = `(Intercept)`) %>%
    mutate(Experiment = unique(orig_data$experiment)) %>%
    select(c(Intercept, Subject, Experiment))
  rownames(data) <- NULL
  data
}

# Custom function to calculate correlations with adjusted p-values in ggpairs
cor_with_p_adjust <- function(data, mapping, method = "pearson", ...) {
  # Extract x and y variables
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)
  
  # Perform correlation test
  test <- cor.test(x, y, method = method)
  
  # Extract p-value and adjust using Bonferroni correction
  p_value <- test$p.value
  # Bonferroni correction: number of comparisons is choose(n, 2)
  p_value_adj <- p.adjust(p_value, method = "BH", n = choose(ncol(data), 2))
  # “holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”
  
  # Create a label for ggally_text
  label <- paste("r = ", round(test$estimate, 2), "\n", "p = ", format.pval(p_value_adj, digits = 2))
  
  # Create ggally_text object
  #ggally_text(label = label, color = ifelse(p_value_adj < 0.05, "red", "black"), ...)
  # also remove grid lines
  ggally_text(label = label, color = ifelse(p_value_adj < 0.05, "red", "black"), ...) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())  # Remove gridlines
}


# LATEX OUTPUTs
output.table.f <- function(data, filename="", thisLabel="", thisCaption=""){
  output <- data %>%
    select(c(`model term`, experiment, sign.fdr)) %>%
    pivot_wider(
      names_from = experiment, 
      values_from = sign.fdr
    ) %>%
    mutate(across(everything(), ~ if_else(is.na(.x), "/", .x))) %>%
    xtable(type="latex",
           label=thisLabel,
           caption=thisCaption)
  
  print(output, # this command saves the xtable to file
        #digits=5,
        include.rownames=FALSE, # row numbers not printed to file
        caption.placement = "top", # caption on top of table
        file = filename)
  filename # it seems that the filename should be printed last for file targets
}

output.table.con <- function(data, filename="", thisLabel="", thisCaption=""){
  output <- data %>%
    select(c(variable, level.1, level.2, experiment, significance)) %>%
    pivot_wider(
      names_from = experiment, 
      values_from = significance
    ) %>%
    mutate(across(everything(), ~ if_else(is.na(.x), "/", .x))) %>%
    xtable(type="latex",
           label=thisLabel,
           caption=thisCaption)
  
  print(output, # this command saves the xtable to file
        #digits=5,
        include.rownames=FALSE, # row numbers not printed to file
        caption.placement = "top", # caption on top of table
        latex.environments = "widestuff", # this uses the widestuff environment which I have designed in latex to adjust the width of the table (move left)
        file = filename)
  filename # it seems that the filename should be printed last for file targets
}

# plot_emm <- function(model, variables){
#   
#   
# }
# 
# 
# find_significant_combinations <- function(features, contrast_df) {
#   # Initialize an empty list to store significant combinations
#   significant_combinations <- list()
#   
#   # Loop over unique experiment combinations
#   for (i in 1:(length(features) - 1)) {
#     for (j in (i + 1):length(features)) {
#       feature1 <- features[i]
#       feature2 <- features[j]
#       
#       # Check if both experiments are present in con$contrast
#       if (any(grepl(feature1, contrast_df$contrast)) && any(grepl(feature2, contrast_df$contrast))) {
#         # Get the row indices where the experiments are found in con$contrast
#         indices1 <- grep(feature1, contrast_df$contrast)
#         indices2 <- grep(feature2, contrast_df$contrast)
#         
#         # Check if there are any significant pairs
#         significant_pairs <- intersect(indices1, indices2)
#         if (length(significant_pairs) > 0) {
#           # Get the corresponding p-values
#           p_values <- contrast_df$p.value[significant_pairs]
#           
#           # Check if any p-value is smaller than 0.05
#           if (any(p_values < 0.05)) {
#             # Add the significant combination to the list
#             significant_combinations[[length(significant_combinations) + 1]] <- c(as.character(feature1), as.character(feature2))
#           }
#         }
#       }
#     }
#   }
#   return(significant_combinations)
# }

# hlm_simulations<- function(data, iterations=1000){
#   data = tar_read(data_eegnet) %>% filter(experiment=="N170")
#   # TODO shuffle labels per sub
#   for (i in 1:iterations){
#     mod_i <- lmer(formula="accuracy ~ hpf + lpf + emc + mac + det + base + ar + (hpf + lpf + emc + mac + det + base + ar | subject)", #experiment + RFX SlOPES
#          control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
#          data = data)
#     # TODO: extract ps or write it into large df
#   }
# }

check_convergence <- function(model){
  if (class(model) == "list"){model <- model[[1]]}
  models <- summary(model)
  
  ## correlations between fixed effects should be not exactly 0, -1 or 1
  corrs <- 
    {if (class(model) %in% c("lmerMod","lmerModLmerTest")) as.matrix(models$vcov) else
    if (class(model) == "lm") models$cov.unscaled} %>%
    cov2cor() %>%
    { .[lower.tri(., diag = FALSE)] }
  
  if (any(corrs %in% c(0, 1, -1))) {
    stop("Some model fixed effect parameter shows a correlation of either 0, 1, or -1!")
  }
  
  ## stdev of fixed effects estimates should not be exactly 0
  if (any(models$coefficients[, "Std. Error"]) == 0){
    stop("Some model fixed effect parameter shows a Std. Error of 0 !")
  }
  
  # output some diagnosticts
  data.frame(CORR = c("ok"),
             SE = c("ok"),
             check_ignore = c(models$coefficients[, "Std. Error"][3]))
}

plot_multiverse_sankey <- function(data){
  data %<>% 
    filter(subject == "sub-001") %>%
    filter(experiment == "N170") %>%
    select(-c(subject, accuracy, experiment)) 
  
  # now change the names of all columns with the replacements
  names(data) <- recode(names(data), !!!replacements)
  
  # make long
  data_long <- data %>%
    make_long(names(data)) %>%
    mutate(node = recode(node, !!!replacements)) %>% # also replace with better names
    mutate(next_node = recode(next_node, !!!replacements))
  
  # reorder factors in node and next_node
  data_long <- data_long %>%
    mutate(node = factor(node, levels = rev(c("None", "ICA", "linear", "False", "interpolate", "reject", "average", "Cz", "P9/P10", "200 ms", "400 ms", "6 Hz", "20 Hz", "45 Hz", "0.1 Hz", "0.5 Hz"))),
           next_node = factor(next_node, levels = rev(c("None", "ICA", "linear", "False", "interpolate", "reject", "average", "Cz", "P9/P10", "200 ms", "400 ms", "6 Hz", "20 Hz", "45 Hz", "0.1 Hz", "0.5 Hz"))))  
  
  ggplot(data_long, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
    geom_sankey(flow.alpha = .6,
                node.color = "gray20") +
    geom_sankey_label(size = 4, color = "white", fill = "gray40") +
    #scale_fill_viridis_d(drop = FALSE) +
    #paletteer::scale_fill_paletteer_d("colorBlindness::paletteMartin") +
    scale_fill_grey() +
    theme_sankey(base_size = 18) +
    labs(x = "") + #, title = "Multiverse" processing step
    theme(legend.position = "none",
          plot.margin=margin(0,0,0,0), #grid::unit(c(0,0,0,0), "mm") # remove white space around plot
          #plot.title = element_text(hjust = .5) # to make it central
          ) +
    scale_x_discrete(position = "top") #+          # Move x-axis to the top
    #coord_cartesian(clip = "off")      
}
