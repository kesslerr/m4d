# Get a list of all installed packages with their versions
installed_pkgs <- installed.packages()

# Extract only the package names and versions
pkg_versions <- installed_pkgs[, c("Package", "Version")]

# Convert to data frame for easier filtering and readability
pkg_versions_df <- as.data.frame(pkg_versions)

# List of packages you want to check
my_pkg_list <- c("colorspace", # nice colormaps
                 "broom",
                 "broom.mixed", # mixed effects functions, e.g. augment for mixed effects models
                 "dplyr", 
                 "ggplot2", 
                 "ggsignif", 
                 "ggpattern", # allows to make patterns into tiles or bars
                 "gtools", # to convert p values into asterisks
                 "ggpubr", # ggplots publication ready, e.g. labeling plots and arranging mutliple plots (ggarrange)
                 "GGally", # e.g. pairwise correlation plots (ggpairs)
                 "ggdist", 
                 "ggthemes", 
                 "ggsankey", # sankey plots
                 "ggview", # scale plots correctly for publication
                 "cowplot", # to overlay plots
                 "pals", # some colorscales
                 "readr", 
                 #"lmerTest", # has p value estimations
                 "lme4", # seems to be 20% faster
                 "emmeans", 
                 "JuliaCall", # use Julia from within R
                 "magrittr", # for double pipe operator %<>%
                 "ggpubr", 
                 "data.table",
                 "tidyverse", 
                 "tidyquant", 
                 "dplyr", # e.g. pipe functions
                 "purrr", 
                 "rstatix", 
                 "stringr", 
                 "tidyr", # e.g. wide to long transform
                 "performance",
                 "circlize", # chord diagram
                 "xtable", # export R tables to latex files
                 "stringr" # string manipulation
) 

# Filter pkg_versions_df based on the list of desired packages
filtered_pkg_versions <- pkg_versions_df[pkg_versions_df$Package %in% my_pkg_list, ]

# Create markdown table
markdown_table <- function(df) {
  # Header row
  cat("| Package | Version |\n")
  cat("|---------|---------|\n")
  
  # Data rows
  apply(df, 1, function(row) {
    cat(sprintf("| %s | %s |\n", row["Package"], row["Version"]))
  })
}

# Call the function to output the markdown table
markdown_table(filtered_pkg_versions)


