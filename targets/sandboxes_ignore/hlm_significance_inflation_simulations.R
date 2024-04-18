library(lmerTest)
library(dplyr)
library(targets)
library(tarchetypes)

## simulations

data = tar_read(data_eegnet) %>% filter(experiment=="N170")


# first one real HLM
model_true <- lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + (1 | subject)", #experiment + RFX SlOPES
              control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
              data = data)

# Function to shuffle hpf values for each subject
shuffle_var <- function(df, var) {
  df %>%
    group_by(subject) %>%
    mutate(!!var := sample(!!rlang::sym(var)))
}

shuffle_var_ordered <- function(df, var) {
  nrows <- data %>% filter(subject=="sub-001") %>% nrow()
  the_order <- order(runif(nrows))
  df %>%
    group_by(subject) %>%
    mutate(!!var := .data[[var]][the_order])
}



# DEBUG
i=1
var="det"
# TODO shuffle labels per sub
ps.1 = list()
ts.1 = list()
#ps.5 = list()

for (i in 1:100){
  
  # Shuffle the hpf values
  #shuffled_data <- shuffle_var(data, var)
  shuffled_data <- shuffle_var_ordered(data, var)
  
  mod_i <- lmer(formula="accuracy ~ hpf + lpf + emc + mac + base + det + ar + ( 1 | subject)", #experiment + RFX SlOPES
       control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)),
       data = shuffled_data)
  
  ps.1[[i]] <- summary(mod_i)$coefficients["detlinear", "Pr(>|t|)"]
  ts.1[[i]] <- summary(mod_i)$coefficients["detlinear", "t value"]
  
  
  #ps.5[[i]] <- summary(mod_i)$coefficients["hpf0.5", "Pr(>|t|)"]
  
  # TODO: extract ps or write it into large df
}

proportion.1 <- mean(ps.1 < 0.05)
print(proportion.1)

threshold_t <- summary(model_true)$coefficients["detlinear", "t value"]
proportion.t1 <- mean(ts.1 < threshold_t)
print(proportion.t1)
proportion.t2 <- mean(ts.1 > threshold_t)
print(proportion.t2)

t_gt_0 <- mean(ts.1 > 0)
t_lt_0 <- mean(ts.1 < 0)
print(t_gt_0)
print(t_lt_0)
# not sure if this makes sense, as the chance level here is quite obvious

#proportion.5 <- mean(ps.5 < 0.05)

# seems no above-chance prediction with unodered shuffled
# however with ordered there is e.g. 0.15 or 0.2 _-> how?



