# MutPPI

MutPPI is a tool for predicting the binding affinity changes (ΔΔG) caused by mutations in protein-protein interactions. This repository contains code and model implementations for predicting both single-point and multi-point mutations using graph neural networks and protein language models. More details can be found in our research work.

***
### System requirements

python (v3.10), torch, torch-geometric, transformers, biopython, pandas, scikit-learn, numpy, scipy, tqdm

***
### Setup of MutPPI

We executed the code under `python=3.10` and `torch=2.4.0+cu118`, we recommend you to use similar package versions.

Install MutPPI: 

```shell
git clone https://github.com/ddd9898/MutPPI.git
cd MutPPI
pip install -r requirements.txt
```

Then, download the checkpoints of ESM-2 650M from the [official link](https://huggingface.co/facebook/esm2_t33_650M_UR50D) or [my copy](https://cloud.tsinghua.edu.cn/d/dae30e6d4ef94707b338/), and put `esm2_t33_650M_UR50D` in the `models` folder.

Download checkpoints from [Link](https://cloud.tsinghua.edu.cn/d/8ac6dfb990b7445aab0f/), unzip files and move them into the `output/checkpoint` folder.

The above installation may take a few hours depending on your internet conditions.

***

### Model Architecture

MutPPI implements three different model architectures:

1. **Baseline Model**: Graph neural network using GIN and GAT layers for structure-based prediction
2. **MutPPI**: Enhanced graph neural network with improved feature extraction
3. **MutPPI plus**: Multi-modal model combining:
   - Graph neural networks for 3D structure information
   - ESM-2 protein language model for sequence information
   - Ensemble prediction with average, min, and max pooling strategies

***

### Usage

#### 1. Single-Point Mutation Prediction

```bash
python predict_singlePoint.py --Model 0 --Rmode 0 --reduction 20 --epoch 1000
python predict_singlePoint.py --Model 1 --Rmode 0 --reduction 20 --epoch 150
```

Parameters:
- `--Model`: Model type (0: MutPPI, 1: MutPPI plus)
- `--Rmode`: Reduction mode (0: all residules, 1: interface residules, 2: nearest residules around the mutation site)
- `--reduction`: When Rmode is 0, it is not applicable. When Rmode is 1, it defines the distance threshold for the binding interface. When Rmode is 2, it indicates the number of amino acids sampled near the mutation site (separately for the mutant protein and the target protein).
- `--epoch`: The training checkpoint to load

#### 2. Multi-Point Mutation Prediction

```bash
python predict_multiPoint.py --Model 1 --Rmode 0 --reduction 20  --epoch 150
```

***

#### 3. Re-testing after data augmentation

- Uncomment line 153 in `predict_multiPoint.py` and comment out line 152.

- Uncomment line 121 in `predict_singlePoint.py` and comment out line 120.

```bash
# S645
python predict_singlePoint.py --Model 1 --Rmode 0 --reduction 20 --epoch 20

# SM_ZEMu、SM595、SM1124
python predict_multiPoint.py  --Model 1 --Rmode 0 --reduction 20 --epoch 20
```


###  Reproduction

#### 1. Training Models

Train baseline model:
```bash
python train_baseline.py --RM 0 --reduction 20
python train_baseline.py --RM 1 --reduction 10
python train_baseline.py --RM 2 --reduction 20
```

Train MutPPI model:
```bash
python train_model.py --RM 0 --Model 0 --reduction 20
python train_model.py --RM 1 --Model 0 --reduction 10
python train_model.py --RM 2 --Model 0 --reduction 20
```

Train MutPPI plus model:
```bash
python train_model.py --RM 0 --Model 1 --reduction 20
```

Train with mutation-path-based data augmentation:
```bash
python train_model_augment.py --Model 0 --reduction 20
python train_model_augment.py --Model 1 --reduction 20
```

#### 2. Cross-Validation on S645

```bash
python train_model_cv.py    --fold 0  --RM 0 
python train_model_cv100.py --fold 0  --RM 0 
python train_model_cv90.py  --fold 0  --RM 0
python train_model_cv70.py  --fold 0  --RM 0
```

### File Structure

```
MutPPI/
├── models/
│   ├── models.py              # Model architectures
│   └── __init__.py
├── utils/
│   ├── dataloader.py          # Single-point mutation data loading
│   ├── dataloader_multi.py    # Multi-point mutation data loading
│   ├── dataloader_CV.py       # Cross-validation data loading
│   ├── evaluate.py            # Evaluation metrics
│   └── __init__.py
├── data/
│   ├── SKEMPI2/              # SKEMPI2 dataset and AB-Bind dataset
│   ├── Graphinity/           # Graphinity cross-validation data
│   └── test/                 # Multiple-mutation test datasets
├── output/
│   ├── checkpoints/          # Saved model weights
│   ├── log/                  # Training logs
│   └── results/              # Prediction results
├── train_*.py                # Training scripts
└── predict_*.py              # Prediction scripts
```

***

### Contact

Feel free to contact [djt20@mails.tsinghua.edu.cn](mailto:djt20@mails.tsinghua.edu.cn) if you have issues for any questions.
