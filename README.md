# lasts (Local Agnostic Subsequence-based Time Series explainer)

## Introduction

`lasts` is a tool designed for explaining time series classifiers. It provides insights into time series models using local agnostic subsequence-based techniques. Due to the reliance on specific older package versions, the current installation method is exclusively through Conda using the provided `environment.yml` file. We aim to offer a more streamlined installation process in future updates.

## Installation

### Prerequisites

- Ensure that [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed on your system.

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/fspinna/lasts.git
   cd lasts
   ```



2. **Create a Conda Environment from the `environment.yml` file:**

   This file lists all the necessary dependencies for `lasts`. By creating a Conda environment from this file, you ensure all dependencies are properly installed and managed.

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment:**

   After creating the environment, activate it with the following command:

   ```bash
   conda activate lasts
   ```

4. **Verify Installation:**

   Post-installation, your terminal prompt should reflect the active `lasts` environment:

   ```bash
   (lasts) user@hostname:~$
   ```

Sure! Let's simplify and streamline the provided code into a quick start guide. 

### Quick Start Guide for `lasts`

Get up and running with the Local Agnostic Subsequence-based Time Series Explainer (`lasts`) with this quick guide.

1. **Import necessary modules**

    ```python
    from lasts.blackboxes.loader import cached_blackbox_loader
    from lasts.datasets.datasets import build_cbf
    from lasts.autoencoders.variational_autoencoder import load_model
    from lasts.utils import get_project_root, choose_z
    from lasts.surrogates.shapelet_tree import ShapeletTree
    from lasts.neighgen.counter_generator import CounterGenerator
    from lasts.wrappers import DecoderWrapper
    from lasts.surrogates.utils import generate_n_shapelets_per_size
    from lasts.explainers.lasts import Lasts
    import numpy as np
    ```

3. **Data preparation**

    Set the random seed and prepare the dataset:
    ```python
    random_state = 0
    np.random.seed(random_state)
    dataset_name = "cbf"
    
    _, _, _, _, _, _, X_exp_train, y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test = build_cbf(n_samples=600, random_state=random_state)
    ```

4. **Load the model and blackbox**

    ```python
    blackbox = cached_blackbox_loader("cbf_knn.joblib")
    encoder, decoder, autoencoder = load_model(get_project_root() / "autoencoders" / "cached" / "vae" / "cbf" / "cbf_vae")
    ```

5. **Preparation for explanations**

    Here, we choose the latent space representation for our instance and set up the counterfactual generator:
    ```python
    i = 0
    x = X_exp_test[i].ravel().reshape(1, -1, 1)
    z_fixed = choose_z(x, encoder, decoder, n=1000, x_label=blackbox.predict(x)[0], blackbox=blackbox, check_label=True, mse=False)
    
    neighgen = CounterGenerator(blackbox, DecoderWrapper(decoder), n_search=10000, ...)

    n_shapelets_per_size = generate_n_shapelets_per_size(X_exp_train.shape[1])
    surrogate = ShapeletTree(random_state=random_state, shapelet_model_kwargs={...})
    ```

6. **Initialize and fit the LASTS explainer**

    ```python
    lasts_ = Lasts(blackbox, encoder, DecoderWrapper(decoder), neighgen, surrogate, verbose=True, binarize_surrogate_labels=True, labels=["cylinder", "bell", "funnel"])

    lasts_.fit(x, z_fixed)
    ```

7. **Generate and visualize explanations**

    ```python
    exp = lasts_.explain()
    
    lasts_.plot("latent_space")
    lasts_.plot("morphing_matrix")
    lasts_.plot("counterexemplar_interpolation")
    lasts_.plot("manifest_space")
    lasts_.plot("saliency_map")
    lasts_.plot("subsequences_heatmap")
    lasts_.plot("rules")
    lasts_.neighgen.plotter.plot_counterexemplar_shape_change()
    ```


## Upcoming Features

We're dedicated to continually refining `lasts`. Look forward to enhancements and a more straightforward installation process in upcoming updates. For issues or feedback, kindly use the repository's "Issues" section.