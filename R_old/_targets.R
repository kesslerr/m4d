# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
# library(tarchetypes) # Load other packages as needed.

# Set target options:
tar_option_set(
  packages = c("tibble", "data.table", "tidyverse", "lmerTest", "ggplot2") # Packages that your targets need for their tasks.
)

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
# tar_source("other_functions.R") # Source other scripts as needed.

# the _targets.R file must always END WITH A LIST!
# Replace the target list below with your own:
list(
  tar_target(name = file, 
             command = "data/eegnet_erpcore.csv",
             format = "file"),
  tar_target(name = data, 
             command = get_data(file))
)
