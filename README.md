# RNA-BAnG

## Installation
Run the following to install a conda environment with the necessary dependencies.
```
conda env create -f environment.yml
```
Next, after the activation of ```rnabang``` environment, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```
## Inference
To run RNA-BAnG inference for protein of your choice use the following command:
```
python experiments/inference_from_cif.py --input_cif /path/to/your/name.cif --nof_samples 20
```
Please make sure that protein structure is unambigous and contained in chain A. AlphaFold predicted structures are suitable by default. 
The command will generate 20 RNA sequences and save them in the ```./inference_outputs/name``` folder. There will be two files: ```seq_na.fasta```, containing final RNA sequences, and ```seq_na_uncut.fasta```, containing corresponding tokens.
