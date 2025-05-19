# Julia setup

All *Julia* functions are called from within the *targets* pipeline in *R*.

Julia is mainly used for speedy linear mixed model fitting.

The deployed package versions used can be found in [Manifest.toml](../env/julia_Manifest.toml) and [Project.toml](../env/julia_Project.toml).


The fitted models are saved in *rds files for import in R.

The models in this folder are from the final reported multiverse. The folder "alternative_order" contains the fitted models of a smaller multiverse, which is reported in the Supplementary Material of the manuscript.
