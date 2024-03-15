# test

df <- tar_read(data_tsum)
estimate_marginal_means_sliding(df, per_exp = TRUE)
  







library(dplyr)
library(purrr)
library(tidyr)
library(data.table)
library(rstatix)

df <- tar_read(data_tsum)

# Group by all variables except tsum and summarize to aggregate tsum values
experiment_value = "MMN"  
variable = "ref" # variable of interest

# mean values
all_results <- data.frame()
for (variable in c("ref","hpf","lpf","base","det","ar","emc","mac")) {
  
  result <- df %>%
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
    )
  all_results <- rbind(all_results, result)
}

# pairwise stats
# TODO loop
# TODO also for all experiments together (average)
variable = "ref"
df2 <- 

all_stats <- data.frame()
for (variable in c("ref","hpf","lpf","base","det","ar","emc","mac")) {
  all_stats <- df %>% 
    filter(experiment == experiment_value) %>% 
    select(c(tsum, !!sym(variable))) %>%
    # rename, so i can use it in the formula
    rename(new_variable_name = !!sym(variable)) %>%
    pairwise_t_test(tsum ~ new_variable_name, 
                    paired = TRUE,
                    p.adjust.method = "BY",
                    pool.sd = FALSE, detailed = TRUE) %>%
    mutate(variable = !!(variable)) %>%
    rbind(all_stats, .)
}




  #result

# Convert tibble to long dataframe
long_df <- unnest(result, cols = tsum_values)

print(long_df)


# If you want to convert the list of tsum values to individual columns:
result <- result %>%
  mutate(tsum_values = map(tsum_values, ~unlist(.x)))

# Print the result
print(result)



# own

df <- tar_read(data_tsum)
# TODO loop across exps
experiment_value = "MMN"  
variable = "ref" # variable of interest

dfexp <- df %>% 
  filter(experiment == experiment_value) %>%
  #melt()
  #pivot_longer(-c(variable, tsum), values_to = "Value", names_to = "Year")
  pivot_longer(
    cols = c("ref","hpf","lpf","base","det","ar","emc","mac"), 
    names_to = "variable",
    values_to = "value"
  )

  #group_by(!!sym(variable))


# TODO: loop across variables
dfexpvar <- dfexp %>% 
  filter(exp)



#for (experiment in unique(df$experiment)){
# todo, also for all experiments together

experiment_value = "MMN"  
dfexp <- df %>% 
  filter(experiment == experiment_value)

long_df <- data.frame()  # Initialize an empty dataframe

for (variable in c("ref","hpf","lpf","base","det","ar","emc","mac")) {
  
  variable = "ref"
  df2 <- dfexp %>% #split(f = variable)
      group_by(!!sym(variable))# %>%
    summarize(tsums = list(tsum))
  df_wide <- data.frame()
  for (i in 1:nrow(df2)){
    row <- df2[i,]
    #print(row[[variable]])
    df_wide[[row[[variable]]]] <- row[["tsums"]]
  }
    #summarize(tsums = list(tsum))
}
  
  #wide <- unnest_wider(variable_df, col=ref, names_sep = "_")
  wide <- as.data.frame(variable_df$tsums)
  wide$ref <- variable_df$ref
  # Reorder the columns
  wide <- wide[, c("ref", 1:384)]
  # Pivot the dataframe from wide to long format and append to long_df
  #long_df <- bind_rows(long_df, pivot_longer(variable_df, cols = tsums, names_to = variable, values_to = "tsum"))
}

print(long_df)