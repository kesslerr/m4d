
## ggpairs with adj correlation p values
## 
## # Install and load the necessary packages

library(GGally)
library(dplyr)

# Sample data
data(mtcars)

# Custom function to calculate correlations with adjusted p-values

# Custom function to calculate correlations with adjusted p-values
cor_with_p_adjust <- function(data, mapping, method = "pearson", ...) { #pearson
  # Extract x and y variables
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)
  
  # Perform correlation test
  test <- cor.test(x, y, method = method)
  
  # Extract p-value and adjust using Bonferroni correction
  p_value <- test$p.value
  # Bonferroni correction: number of comparisons is choose(n, 2)
  p_value_adj <- p.adjust(p_value, method = "BH", n = choose(ncol(data), 2))
  
  # Create a label for ggally_text
  label <- paste("r = ", round(test$estimate, 2), "\n", "p = ", format.pval(p_value_adj, digits = 2))
  
  # Create ggally_text object
  #ggally_text(label = label, color = ifelse(p_value_adj < 0.05, "red", "black"), ...)
  # also remove grid lines
  ggally_text(label = label, color = ifelse(p_value_adj < 0.05, "red", "black"), ...) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())  # Remove gridlines
}


ggpairs(wide_data, 
        upper = list(continuous = wrap(cor_with_p_adjust))) +
        #upper = list(continuous = "cor")) + 
  #labs(title="Random Intercept Correlation Between Experiments") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 


