# Kaggle sessions

These are the notebooks as they ran on Kaggle, one per session. Each one writes `run_experiments.py` into the working folder and runs a few configurations. `01_training.ipynb` and `02_analysis.ipynb` one folder up do the same work in two notebooks and are easier to rerun.

Kaggle limits how long a GPU session can run, so the 55 training runs were spread over many sessions. When a session ran out of time, its remaining runs were moved to a smaller notebook (for example 02 became 02a and 02b). That is why some configurations appear in more than one notebook. A run is skipped when its results file already exists, so no run was counted twice; `08_analysis` keeps only the first copy of each run name.

`07_oversampling_other_architectures` tests oversampling on DenseNet121, EfficientNet-B0 and EfficientNetV2-S. These runs are not part of the paper and are not in `outputs/`. `08_analysis_first_version` is an earlier analysis notebook that was replaced by `08_analysis`. `colab_resnet50_A4` runs ResNet50 A4 on Google Colab instead of Kaggle.
