# How EEG preprocessing shapes decoding performance
Multiverse for Decoding (m4d).

R. Kessler et al. (2024)

See preprint [here](https://doi.org/10.48550/arXiv.2410.14453) 

EEG preprocessing varies widely between studies, but its impact on stimulus classification performance remains poorly understood. To address this gap, we analyzed seven experiments with 40 participants drawn from the public ERP CORE dataset. We systematically varied key preprocessing steps, such as filtering, referencing, baseline interval, detrending, and multiple artifact correction steps. Then we performed trial-wise binary classification (i.e., decoding) using neural networks (EEGNet), or time-resolved logistic regressions. Our findings demonstrate that preprocessing choices influenced decoding performance considerably. All artifact correction steps reduced decoding performance across all experiments and models, while higher high-pass filter cutoffs consistently enhanced decoding. For EEGNet, baseline correction further improved performance, and for time-resolved classifiers, linear detrending and lower low-pass filter cutoffs were beneficial. Other optimal preprocessing choices were specific for each experiment. The current results underline the importance of carefully selecting preprocessing steps for EEG-based decoding. If not corrected, artifacts facilitate decoding but compromise conclusive interpretation.


# Structure of this repository

Subfolders will contain READMEs which are more specific.

General structure adapted from [cookiecutter](https://github.com/drivendata/cookiecutter-data-science):
```

├── README.md          <- The top-level README.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── env                <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `conda export >> env.json`
│
├── julia              <- Julia scripts.
│
├── manuscript         <- Manuscript submodule pointing to a different repository synced with Overleaf
│
├── models             <- Trained models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks and similar with one-off analyses
│
├── plots              <- Plots (other plots are directly plotted into the manuscript folder)
│
├── poster             <- Conference posters
│
├── presentation       <- Project presentations
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── src                <- Python source code for use in this project.
│
├── targets            <- R (targets) Pipeline.





