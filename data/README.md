## Data Description

The repository includes several datasets for training and evaluation:

#### Training Data
- **SKEMPI2**: Large-scale protein-protein interaction mutation dataset
  - S4169  single-point mutations: `./data/SKEMPI2/S4169.csv`
#### Test Data
- **Single Mutation Test Set**: 
	- S645 antibody-antigen single-mutation data (AB-Bind dataset):  `./data/SKEMPI2/AB-Bind-Database/AB_645.csv`
- **Multiple Mutation Test Sets**: 
  - SM_ZEMu: `./data/test/multiple/SM_ZEMu.csv`
  - SM595: `./data/test/multiple/SM595.csv`
  - SM1124: `./data/test/multiple/SM1124.csv`
- **Graphinity's Cross-validation on S645**: `./data/Graphinity/Experimental_ddG_645_+Reverse_Mutations_+Non_Binders/`
#### Data For Analysing
- `SKEMPI2_MultiMuts.csv`: Multi-point mutations derived from SKEMPI2.

Each dataset includes:
- PDB structure information
- Mutation annotations (e.g., "A16D" for Alanine to Aspartic acid at position 16)
- Experimental ΔΔG values
- 3D coordinate files in PDB format

***
