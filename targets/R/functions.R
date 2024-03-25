get_preprocess_data <- function(file) {
  data <- read_csv(file, col_types = cols())
  # new, preprocess already
  data$hpf <- factor(data$hpf, levels = c("None", "0.1", "0.5"))
  data$lpf <- factor(data$lpf, levels = c("None", "6", "20", "45")) # TODO: put None to last level, this will however rerun the pipeline
  data$ref <- factor(data$ref, levels = c("average", "Cz", "P9P10"))
  data$emc <- factor(data$emc, levels = c("None", "ica"))
  data$mac <- factor(data$mac, levels = c("None", "ica"))
  data$base <- factor(data$base, levels = c("200ms", "400ms"))
  data$det <- factor(data$det, levels = c("offset", "linear"))
  data$ar <- factor(data$ar, levels = c("FALSE", "TRUE"))
  data$experiment <- factor(data$experiment, levels = c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3", "LRP_6-9", "LRP_10-11", "LRP_12-13", "LRP_14-17", "LRP_18+", "6-9", "10-11", "12-13", "14-17", "18+"))
  data$dataset <- factor(data$dataset)
  data
}

# factorize <- function(data) {
#   data$hpf <- factor(data$hpf, levels = c("None", "0.1", "0.5"))
#   data$lpf <- factor(data$lpf, levels = c("None", "6", "20", "45"))
#   data$ref <- factor(data$ref, levels = c("average", "Cz", "P9P10"))
#   data$emc <- factor(data$emc, levels = c("None", "ica"))
#   data$mac <- factor(data$mac, levels = c("None", "ica"))
#   data$base <- factor(data$base, levels = c("200ms", "400ms"))
#   data$det <- factor(data$det, levels = c("offset", "linear"))
#   data$ar <- factor(data$ar, levels = c("FALSE", "TRUE"))
#   data$experiment <- factor(data$experiment)
#   data
# }

estimate_marginal_means <- function(data, variables){
  data_list <- list()
  for (variable in variables) {
    average_data <- data %>%
      group_by(subject, !!sym(variable)) %>%
      summarize(accuracy = mean(accuracy))# %>%
    #ungroup()
    average_data$variable <- names(average_data)[2]
    average_data$factor <- average_data[[variable]]
    average_data[[variable]] <- NULL
    data_list <- append(data_list, list(average_data))
  }
  # Concatenate the data frames in the list
  bind_rows(data_list)
}

estimate_marginal_means_sliding <- function(data, per_exp = FALSE){
  # mean values
  if (per_exp == TRUE){
    experiments <- c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")
  } else {
    experiments <- c("dummy")
  }
  all_results <- data.frame()
  for (experiment_value in experiments){
    for (variable in c("ref","hpf","lpf","base","det","ar","emc","mac")) {
      
      result <- data %>%
        {if(per_exp==TRUE) filter(., experiment == experiment_value) else . } %>%
        #filter(experiment == experiment_value) %>%
        group_by(experiment, !!sym(variable)) %>% # TODO, add experiment as first grouping variable for each 
        summarize(tsum_values = list(tsum)) %>%
        unnest(cols = tsum_values) %>%
        group_by(!!sym(variable)) %>%
        summarize(tsum=mean(tsum_values)) %>%
        pivot_longer(
          cols = c(variable), 
          names_to = "variable",
          values_to = "level"
        ) %>%
        {if(per_exp==TRUE) mutate(., experiment = experiment_value) else . }
        
      all_results <- rbind(all_results, result)
    }
  }
  all_results
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
  ref = c('P9P10', 'P9P10', 'P9P10', 'average', 'P9P10', 'P9P10', 'P9P10'),
  hpf = c('0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1'),
  lpf = c('None', 'None', 'None', 'None', 'None', 'None', 'None'),
  emc = c('ica', 'ica', 'ica', 'ica', 'ica', 'ica', 'ica'),
  mac = c('ica', 'ica', 'ica', 'ica', 'ica', 'ica', 'ica'),
  base = c('200ms', '200ms', '200ms', '200ms', '200ms', '200ms', '200ms'),
  det = c('offset', 'offset', 'offset', 'offset', 'offset', 'offset', 'offset'),
  ar = c('TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE')
)

timeresolved_plot <- function(data){
  data_fp <- semi_join(data, luckfps, 
                       by = c("experiment", "ref", "hpf", "lpf", "emc", "mac", "base", "det", "ar"))
  # TR-Decoding with points as significance markers
  ggplot(data_fp, aes(x = times, y = `balanced accuracy`)) +
    geom_line() +
    geom_hline(yintercept=0.5, linetype="solid") +
    geom_vline(xintercept=0, linetype="dashed") +
    geom_point(data=filter(data_fp, significance=="TRUE"),
               aes(x=times, y=0.48),
               color="aquamarine4",
               size=1
    ) +

    facet_wrap(experiment~., scales = "free_x", ncol=1) +
    scale_x_continuous(breaks = seq(-8, 8, by = 2)/10, 
                       labels = seq(-8, 8, by = 2)/10) +
    labs(x="Time", y="Accuracy", title="Examplary Sliding Window Results - 1 UNIVERSE")
  
  
}

# ecdf plot with the best pipeline(s) marked for each experiment
ecdf <- function(data){
  
  best_data = data.frame()
  for (experiment_val in c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")){
    newdata <- data %>%
      #group_by(ref, hpf, lpf, emc, mac, base, det, ar) %>%
      #summarize(tsum = mean(tsum)) %>%
      filter(experiment==experiment_val) %>%
      mutate(performance = # TODO: adjust to the one or many values which are best according to the statistics you chose
               (ar==FALSE) & 
               #ref=="P9P10"
               #base=="400ms" & 
               (det=="linear") & 
               (emc=="None") & 
               (mac=="None") & 
               (hpf==0.5) & 
               (lpf==6)
      ) %>% # if these conditions are met, then write TRUE, else FALSE
      #ungroup() %>% # because row names etc are not correct in a grouped df
      arrange(tsum) %>% # sort by ascending tsum (if not, write desc(tsum))
      mutate(idx = as.numeric(row.names(.))/1152) # index column
    best_data = rbind(best_data, newdata)
  }
  
  ggplot(best_data, aes(x=tsum)) +
    geom_hline(data =filter(best_data, performance==TRUE),
               aes(yintercept = idx),
               color="darkgrey") +
    stat_ecdf() +
    facet_wrap(experiment ~., scales = "free_x")
}


paired_tests <- function(data, study="ERPCORE"){
  # if we are in the MIPDB data, then for the experiment variable can not be grouped for tests,
  # therefore, delete the experiment variable, and make an unpaired test for it
  # 1. only try by removing experiment in MIPDB
  
  #if (study=="MIPDB"){
  #  data_exp <- data %>% filter(variable == "experiment") # NEW
  #  data <- data %>% filter(variable != "experiment")
  #}
  
  results <- data %>%
    group_by(variable) %>%
    pairwise_t_test(accuracy ~ factor, paired = TRUE, p.adjust.method = "BY",
                    pool.sd = FALSE, detailed = TRUE)

  #if (study=="MIPDB"){
  #  results_exp <- data_exp %>%
  #    group_by(variable) %>%
  #    pairwise_t_test(accuracy ~ factor, paired = FALSE, p.adjust.method = "BY",
  #                    pool.sd = FALSE, detailed = TRUE) %>%
  #    select(-estimate1, -estimate2)
  #  results <- rbind(results, results_exp)
  #}
  
  results
}




# rename variables
replacements <- list(
  "hpf" = "high pass (Hz)",
  "lpf" = "low pass (Hz)",
  "ref" = "reference",
  "ar" = "autoreject",
  "mac" = "muscle artifact corr.",
  "emc" = "eye movement corr.",
  "base" = "baseline corr.",
  "det" = "detrending",
  "0.1" = "0.1Hz",
  "0.5" = "0.1Hz",
  "6" = "6Hz",
  "20" = "20Hz",
  "45" = "45Hz",
  "FALSE" = "False",
  "TRUE" = "True",
  "ica" = "ICA",
  "P9P10" = "P9 / P10"
)

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
    labs(title = title,
         x="Experiment",
         y=DV)
  if (DV=="Accuracy"){
    p <- p + geom_hline(yintercept=0.5, lty="dashed")
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


filter_experiment <- function(data){
  experiments = unique(data$experiment)
  data_list = list()
  for (experiment in experiments){
    data_list[[experiment]] <- data[experiment == experiment, ]
  }
  data_list
}



## HLM and EMM


#hlm <- function(data, formula) {
#  lmer(formula, data = data)
#}

est_emm <- function(model, variables){
  # DEBUG
  #variables = c("ref", "hpf","lpf","base","det","ar","emc","mac","experiment")
  
  means = data.frame()
  contra = data.frame()
  
  for (variable in variables){
    # MAIN EFFECTS (1 factor)
    emm <- emmeans(model, 
                   specs = formula(paste0(c("pairwise ~ ",variable))), 
                   #lmerTest.limit = 322560,
                   #pbkrtest.limit = 322560) # to not have inf df
    )
    
    # get means
    dfw <- emm$emmeans %>% 
      as.data.frame() # leaving out contrasts for now
    dfw$variable <- names(dfw)[1]
    names(dfw)[1] <- "level"
    dfw <- dfw[, c(7, 1, 2)]  # CAVE: the SD/CIs can not be used (see warning and values), therefore cutting them
    means <- rbind(means, dfw)
    
    # get contrasts
    dfc <- emm$contrasts %>% 
      as.data.frame() %>% # leaving out contrasts for now
      mutate(variable = variable) %>%
      separate(contrast, c("level.1", "level.2"), " - ")
    dfc <- dfc[, c(8, 1, 2, 3, 4, 5, 6, 7)]
    contra <- rbind(contra, dfc)
    
  }
  # significance asterisks
  contra <- contra %>% mutate(significance = stars.pval(.$p.value) )
  
  return(list(means,contra))
  
}

# ungroup targets across all branches
ungrouping <- function(input){
  data = data.frame()
  i <- 0
  for (experiment in c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")){
    i <- i + 1
    #tmp <- tar_read(eegnet_HLM_exp_emm_means, branches=i)[[1]]
    tmp <- input[[i]]
    tmp[["experiment"]] <- experiment
    data <- rbind(data, tmp)
  }
  data
}

# concatenate experiment and whole 
combine_single_whole <- function(single, whole){
  whole$experiment = "ALL"
  rbind(whole, single)
}

# heatmap of emms
heatmap <- function(data){
  data <- data %>% 
    # Apply replacements batchwise across all columns
    mutate(variable = recode(variable, !!!replacements)) %>%
    # delete the experiment compairson in the full data
    filter(!(experiment == "ALL" & variable == "experiment")) %>% 
    # center around zero for better comparability
    group_by(experiment) %>%
    mutate(emmean = (emmean / mean(emmean) - 1) * 100 ) # now it is percent

  ggplot(data, aes(y = 0, x = level, fill = emmean)) +
    geom_tile() +
    facet_grid(experiment~variable, scales="free") +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank()) +
    scale_fill_continuous_diverging(palette = "Blue-Red 3", 
                                    l1 = 45, # luminance at endpoints
                                    l2 = 100, # luminance at midpoints
                                    p1 = .9, 
                                    p2 = 1.2) +
    labs(x="processing level",
         y="",
         fill="delta\nfrom\nmarginal\nmean\n(%)")  
        # Percentage marginal mean discrepancy
        # Distance from average (in %)
        # Percent above/below average
}

qqplot.data <- function (model, data="", title="") # argument: vector of numbers
{
  if (is.data.frame(data)){
    title=unique(data$experiment)
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
#     mod_i <- lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + (hpf + lpf + emc + mac + base + det + ar | subject)", #experiment + RFX SlOPES
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
    {if (class(model) == "lmerMod") models$vcov else
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
