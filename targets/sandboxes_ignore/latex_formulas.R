
# equationexport latex

library(equatiomatic)

# LM

model <- tar_read(sliding_LM)

# Give the results to extract_eq
extract_eq(model,
           intercept = "beta", # make it beta_0 instead of alpha
           #var_colors=c("blabla"="blue"),
           #var_subscript_colors=c("blabla"=darkred)",
           wrap=TRUE, # this and next line wrap the function
           terms_per_line = 5,
           # swap_var_names
           
           )

# LMM

model <- tar_read(eegnet_HLM_exp, branches=1)[[1]]

# Give the results to extract_eq
extract_eq(model,
           #intercept = "beta", # make it beta_0 instead of alpha
           #var_colors=c("blabla"="blue"),
           #var_subscript_colors=c("blabla"=darkred)",
           wrap=TRUE, # this and next line wrap the function
           terms_per_line = 5,
           # swap_var_names
           
)
# HERE IS AN ERROR






# simpler model
data(mtcars)

model1 <- lm(mpg ~cyl + vs + am, data = mtcars)
extract_eq(mod1)
