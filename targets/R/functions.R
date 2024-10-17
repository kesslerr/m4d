library(JuliaCall)
options(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")
#julia_executable <- "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/julia"
julia_setup(JULIA_HOME = "/Users/roman/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/bin/")

# colormap from the numerosity paper
colors_dark <- c("#851e3e", "#4f7871", "#3c1d85") # red, green, purple "#537d7d", 
colors_light <- c("#f6eaef", "#f2fefe", "#9682c0")

#rkcolors  = c("#792427",
#              "#54828e",
#              "#d1bda2")


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


# raw accuracies in a raincloud plot
raincloud_acc <- function(data, title = ""){
  names(data)[1] <- str_to_title(names(data)[1])
  DV <- names(data)[1]
  if (DV == "Tsum"){
    DV_label <- "T-sum"
  } else {
    DV_label <- DV
  }
  
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
      scale = 0.8, ##  <(**new parameter**),<
      
      # new
      fill = "#565656",  # Set fill color to black (or another color of your choice)
      alpha = 0.8      # Set alpha for opacity (lower values for more transparency)
    ) +
    geom_boxplot(
      width = 0.12,
      # removing outliers
      outlier.color = NA,
      alpha = 0.5
    ) +
    labs(title = title,
         x="Experiment",
         y=DV_label)
  if (DV=="Accuracy"){
    p <- p + geom_hline(yintercept=0.5, lty="dashed")
    p <- p + lims(y=c(0.46,NA))
  }
  p
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
    geom_tile(width = 1) + # 
    #theme_void() +
    #geom_text(aes(label = sprintf("%.1f", emmean)), size = 3) + # Add text labels with one decimal place
    geom_text(aes(label = sprintf("%+.1f", emmean)), size = 3) + # Add text labels with one decimal place and + sign for positives
    facet_grid(experiment~variable, scales="free") +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.background = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          
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
    labs(x="Preprocessing step",
         y="",
         #fill="% change\naccuracy")  
         fill="% Deviation from\nmarginal mean")  
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





plot_multiverse_sankey <- function(data){
  # DEBUG
  #data <- tar_read(data_eegnet)
  
  data %<>% 
    filter(subject == "sub-001") %>%
    filter(experiment == "N170") %>%
    select(-c(subject, accuracy, experiment)) 
  
  # now change the names of all columns with the replacements
  names(data) <- recode(names(data), !!!replacements)

  data <- data %>%
    mutate(ocular = recode(ocular, !!!replacements)) %>%
    mutate(muscle = recode(muscle, !!!replacements)) %>%
    mutate(`low pass` = recode(`low pass`, !!!replacements)) %>%
    mutate(`high pass` = recode(`high pass`, !!!replacements)) %>%
    mutate(reference = recode(reference, !!!replacements)) %>%
    mutate(detrending = recode(detrending, !!!replacements)) %>%
    mutate(baseline = recode(baseline, !!!replacements)) %>%
    mutate(autoreject = recode(autoreject, !!!replacements))
    
  ## new: define example forking paths by adding * or ** (N170) to the steps
  # in column ocular, change ica to ica*
   data <- data %>%
     mutate(ocular = recode(ocular, "ICA" = "ICA*")) %>%
     mutate(muscle = recode(muscle, "ICA" = "ICA*")) %>%
     mutate(`low pass` = recode(`low pass`, "None" = "None*")) %>%
     mutate(`high pass` = recode(`high pass`, "0.1 Hz" = "0.1 Hz*")) %>%
     mutate(reference = recode(reference, "average" = "average*")) %>%
     mutate(reference = recode(reference, "P9/P10" = "P9/P10*")) %>%
     mutate(detrending = recode(detrending, "None" = "None*")) %>%
     mutate(baseline = recode(baseline, "200 ms" = "200 ms*")) %>%
     mutate(autoreject = recode(autoreject, "interpolate" = "interpolate*"))
     
  
  # make long
  data_long <- data %>%
    make_long(names(data)) #%>%
    #mutate(node = recode(node, !!!replacements)) %>% # also replace with better names
    #mutate(next_node = recode(next_node, !!!replacements))
  
  # reorder factors in node and next_node
  data_long <- data_long %>%
    mutate(node = factor(node,           levels = rev(c("None", "None*", "ICA*", "linear", "False", "interpolate*", "reject", "average*", "Cz", "P9/P10*", "200 ms*", "400 ms", "6 Hz", "20 Hz", "45 Hz", "0.1 Hz*", "0.5 Hz"))),
           next_node = factor(next_node, levels = rev(c("None", "None*", "ICA*", "linear", "False", "interpolate*", "reject", "average*", "Cz", "P9/P10*", "200 ms*", "400 ms", "6 Hz", "20 Hz", "45 Hz", "0.1 Hz*", "0.5 Hz"))))  
    # mutate(node = factor(node,           levels = rev(c("None", "ICA", "linear", "False", "interpolate", "reject", "average", "Cz", "P9/P10", "200 ms", "400 ms", "6 Hz", "20 Hz", "45 Hz", "0.1 Hz", "0.5 Hz"))),
    #        next_node = factor(next_node, levels = rev(c("None", "ICA", "linear", "False", "interpolate", "reject", "average", "Cz", "P9/P10", "200 ms", "400 ms", "6 Hz", "20 Hz", "45 Hz", "0.1 Hz", "0.5 Hz"))))  
  
  
  
  # test column title zeilenumbruch
  data_long <- data_long %>%
    mutate(x = recode(x, "ocular" = "ocular\nartifact\ncorrection")) %>%
    mutate(x = recode(x, "muscle" = "muscle\nartifact\ncorrection")) %>%
    mutate(x = recode(x, "low pass" = "low\npass\nfilter")) %>%
    mutate(x = recode(x, "high pass" = "high\npass\nfilter")) %>%
    mutate(x = recode(x, "baseline" = "baseline\ncorrection")) %>%
    mutate(x = recode(x, "autoreject" = "autoreject\nversion"))
  
  p1 <- ggplot(data_long, 
               aes(x = x, next_x = next_x, node = node, next_node = next_node, label = node)) + # fill = factor(node), 
    geom_sankey(flow.alpha = .4,
                node.color = "gray30",
                fill = "grey30") +
    geom_sankey_label(size = 4, color = "white", fill = "gray20", fontface = "bold") +
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
  
  # new: try manipulate colors manually to indicate reference level
  #p1 + scale_fill_manual(values = c('None'    = "black",
  #                                  'average' = "black"))
  p1
}
