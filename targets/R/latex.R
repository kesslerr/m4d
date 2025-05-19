
# R1: changed fdr correction to unc in F-tests

# Create a custom function to perform replacements within strings (eg emc:mac)
replace_with_list <- function(string, replacements) {
  for (pattern in names(replacements)) {
    string <- str_replace_all(string, pattern, replacements[[pattern]])
  }
  return(string)
}

# LATEX OUTPUTs
output.table.f <- function(data, filename="", thisLabel="", thisCaption=""){
  output <- data %>%
    select(c(`model term`, experiment, sign.unc)) %>%
    pivot_wider(
      names_from = experiment, 
      values_from = sign.unc
    ) %>%
    mutate(across(everything(), ~ if_else(is.na(.x), "/", .x))) %>%
    mutate(`model term` = sapply(`model term`, replace_with_list, replacements = replacements)) %>% # replace the variable names with the full names
    xtable(type="latex",
           label=thisLabel,
           caption=thisCaption)
  
  print(output, # this command saves the xtable to file
        #digits=5,
        include.rownames=FALSE, # row numbers not printed to file
        caption.placement = "top", # caption on top of table
        latex.environments = "widestuff", # this uses the widestuff environment which I have designed in latex to adjust the width of the table (move left)
        table.placement = "!htp",
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
    mutate(variable = recode(variable, !!!replacements)) %>%
    mutate(level.1 = recode(level.1, !!!replacements)) %>%
    mutate(level.2 = recode(level.2, !!!replacements)) %>%
    xtable(type="latex",
           label=thisLabel,
           caption=thisCaption)
  
  print(output, # this command saves the xtable to file
        #digits=5,
        include.rownames=FALSE, # row numbers not printed to file
        caption.placement = "top", # caption on top of table
        latex.environments = "widestuff", # this uses the widestuff environment which I have designed in latex to adjust the width of the table (move left)
        table.placement = "!htp",
        file = filename)
  filename # it seems that the filename should be printed last for file targets
}
