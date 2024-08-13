

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
