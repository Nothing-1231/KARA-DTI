# KARA-DTI

KARA-DTI is a deep learning framework for predicting drug–target interactions. The model integrates multi-source biochemical features, a learnable router-enhanced graph network, and a dual-encoder architecture to capture complex cross-modal relationships between drugs and proteins.

---

## Requirements

The implementation is based on Python and PyTorch.

Key dependencies include:

- python == 3.10
- torch == 2.2.0
- torch-geometric == 2.6.1
- numpy == 1.26.4
- scipy == 1.15.2
- pandas == 2.2.3
- scikit-learn == 1.5.2
- rdkit == 2023.9.6
- transformers == 4.50.3
- tqdm == 4.67.1
- biopython == 1.83

### Additional Tools

- **CD-HIT**: Required for protein sequence clustering in cold-start splits.
  Installation: `conda install -c bioconda cd-hit` or `sudo apt-get install cd-hit`

## Dataset

The datasets used in this study are publicly available benchmark datasets, including:

- KIBA  
- C.elegans  
- BindingDB  
- Human  

Due to GitHub file size limitations, the datasets are not included in this repository.

This script generates four data splitting strategies:

Split	Description

--split 0 (S1)	Random split (8:1:1 train/val/test), repeated 20 times

--split 1 (S2)	Cold-drug: unseen drugs in test set

--split 2 (S3)	Cold-target: unseen targets with CD-HIT 40% sequence clustering

--split 3 (S4)	Cold-pair: both drugs and targets unseen with CD-HIT clustering

Users should download the original datasets and place them in the following directory structure:

data/
    dataset_name/
        drug.txt
        protein.txt
        interaction.txt

---

## Running the Model

To train the model, run:python main.py --dataset KIBA --split 0

Train with 20 independent runs (as used in the paper)：python main.py --dataset KIBA --split 0 --num_runs 20
