# How EEG preprocessing shapes decoding performance
Working title: Multiverse 4 Decoding (m4d)

**Kessler et al., 2024**, How EEG preprocessing shapes decoding performance. *Arxiv*. [doi.org/10.48550/arXiv.2410.14453](doi.org/10.48550/arXiv.2410.14453)

See preprint [here](https://doi.org/10.48550/arXiv.2410.14453) 

EEG preprocessing varies widely between studies, but its impact on stimulus classification performance remains poorly understood. To address this gap, we analyzed seven experiments with 40 participants drawn from the public ERP CORE dataset. We systematically varied key preprocessing steps, such as filtering, referencing, baseline interval, detrending, and multiple artifact correction steps. Then we performed trial-wise binary classification (i.e., decoding) using neural networks (EEGNet), or time-resolved logistic regressions. Our findings demonstrate that preprocessing choices influenced decoding performance considerably. All artifact correction steps reduced decoding performance across all experiments and models, while higher high-pass filter cutoffs consistently enhanced decoding. For EEGNet, baseline correction further improved performance, and for time-resolved classifiers, linear detrending and lower low-pass filter cutoffs were beneficial. Other optimal preprocessing choices were specific for each experiment. The current results underline the importance of carefully selecting preprocessing steps for EEG-based decoding. If not corrected, artifacts facilitate decoding but compromise conclusive interpretation.


# Structure of this repository

Subfolders will contain READMEs which are more specific.

**Note: The multiverse-preprocessed comprises >15 TB. It will be shared on a suitable data sharing platform at a later stage.**

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

```

# Environments / Packages


The conda environment is saved in folder [env](/env). All python/bash/slurm scripts are found in [src](/src).

The system architecture and hardware details of the HPC used for all *Python* and *Bash* scripts  with *SLURM* job scheduling system can be found in [MPCDF RAVEN user guide](https://docs.mpcdf.mpg.de/doc/computing/raven-details.html).

The *R* environment, which is used in a *targets* pipeline and all related processing scripts are found in [targets](/targets).

The *Julia* environment for LMM fitting are found in [julia](/julia).

The system architecture and hardware details of the Mac used to process the *targets* pipeline in *R* and *Julia* can be found [here](https://support.apple.com/en-us/111893). A 16 GB RAM version was used.

# Run analyses

The following is done on a HPC cluster with SLURM job scheduling system and the conda environment set-up.

## Multiverse preprocessing and machine learning model fitting

Download the ERP CORE data for all participants and experiments.  :hourglass_flowing_sand: several minutes to hours, depending on bandwidth  

```
python3 src/0-download.py
```

Prepare the data   :hourglass_flowing_sand: <1h
- rearrange trigger values
- rename annotations
- get times
- resample to 256 Hz
- calculate artifical EOG channels
- set montage

```
python3 src/1-pre-multiverse.py
```

Run multiverse preprocessing: For each experiment and participant, preprocess the raw data using >2500 different preprocessing pipelines.  :hourglass_flowing_sand: 24h per participant and experiment

```
bash src/2-multiverse.sh
```

Calculate evoked responses, visualize particularly for an example forking path.  :hourglass_flowing_sand: <1h

```
python3 src/3-evoked.py
```

Run decoding for each forking path, participant, and experiment:
- EEGNet decoding  :hourglass_flowing_sand: 24h per participant and experiment
- Time-resolved decoding  :hourglass_flowing_sand: <1h per participant and experiment

```
bash src/4a-eegnet.sh
bash src/4b-sliding.sh
```

Aggregate EEGNet results for analysis in R/targets.  :hourglass_flowing_sand: <1h
```
python src/5a-aggregate_results.py
```

Aggregate time-resolved results on group level for analysis in R/targets, and visualize for example forking path.   :hourglass_flowing_sand: <1h
```
python src/5b-sliding_group.py
```

## Fitting Linear Mixed Models in Julia

From a terminal with *Julia* installed based on the environment.    :hourglass_flowing_sand: <24h 

```
julia julia/pretarget_model_fitting_en.jl
julia julia/pretarget_model_fitting_tr.jl
```

The model fitting in *Julia* is an inifinite times faster than in *R*, especially for large models and data sets.
The bottleneck however is the conversion from a *Julia* LMM object to an *R* LMM object, which takes a few hours per model.

The present steps were performed before the *targets* pipeline to prevent computationally intensive steps to run after pipeline invalidation. Other, less intensive steps run in Julia are performed later via *targets* pipeline, but are not for the main results but refer to calculations done for the appendix. 

## Modeling the impact of preprocessing on decoding performance

The following is performed within an *R* *targets* pipeline, with access to *Julia* language. From within RStudio, source ```targets/renv/activate.R``` and ```targets/_targets.R```. *_targets.R* contains the entire pipeline.

The pipeline (and the status of each node) can be visualized using
```
tar_visnetwork()
```

The complete pipeline is run using  :hourglass_flowing_sand: <2h
```
tar_make()
```

The resulting plots are directly plotted into the *manuscript* folder (git submodule).

# License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
