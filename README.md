# SPN\_LSM

This repository provides code for the implementation of our paper **Counterfactual Explanations in Medical Imaging:
Exploring SPN-Guided Latent Space Manipulation**, submitted to 24th International Workshop on Data Mining in Bioinformatics (BioKDD25). The code includes preprocessing datasets, training the model, generating counterfactuals, and evaluating their quality.

## Repository Structure

* **preprocessing.py**: Script for dataset preprocessing.
* **main\_pipeline.py**: Code for training the SPN-VAE model.
* **cf\_creation.py**: Script for generating counterfactuals for trained models.
* **main\_analysis.py**: Code for evaluating the quality of generated counterfactuals.
* **example\_plot.png**: Example visualization of results.

## Getting Started

### Prerequisites

* Python 3.8+
* Required libraries can be installed via:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Run the following to preprocess the dataset and enter your Hugging Face access token:

```bash
python preprocessing.py
```

### 2. Model Training

To train the SPN-VAE model, use:

```bash
python main_pipeline.py --le_warmup 0 --load_pretrain_model 0 --gauss_std 0.2 --save_path 'test_model' --loss_weights 10.0 0.1 5.0 --dropout 0.0 --epochs 101 --fine_tune_its 0 --use_add_info 1 --use_VAE 1 --VAE_debug 1 --latent_dim 64 --separate_view 0 --learning_rate 0.0001 --fine_tune_rate 0.0001 --batch_size 50
```

### 3. Counterfactual Generation

For creating counterfactuals with the trained model:

```bash
python cf_creation.py
```

### 4. Counterfactual Evaluation

Evaluate the generated counterfactuals using:

```bash
python main_analysis.py
```


## Contact

For any questions, please contact siekiera@uni-mainz.de .
