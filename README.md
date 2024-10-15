# PATHpre
PATHpre is a a Large-scale Biophysical Sampling Augmented Deep Learning model that used to predict the transition pathway of proteins. 

Please kindly cite: "Exploring Protein Conformational Changes Using a Large-scale Biophysical Sampling Augmented Deep Learning Strategy", Advanced Science, 2024

![fig3_5](https://github.com/user-attachments/assets/068f8173-066f-45ec-bf30-6380fa6168cd)

# Environment
numpy == 1.23.1

torch == 1.12.1

python == 3.8.0

mdtraj == 1.9.9

# Getting started
Example: G-protein
To predict one high energy state (~transition state) from two static states (pdb1 & pdb2):
```shell
python3 PathPre.py --pdb1 2lhc.pdb --pdb2 2lhd.pdb --cut 1.0
```
The output will be a high-energy state contact map, named HES.npy, generated with a cutoff of 1.0 nm. A 1.0 nm cutoff represents Dij = 0.5 in our paper. The recommendated cutoff is 0.8 ~ 1.2 nm depending on proteins.

Example: RfaH
To predict the allosteric path from two static states (pdb1 & pdb2):
```shell
python3 PathPre.py --pdb1 2oug.pdb --pdb2 2lcl.pdb --pre_path --cut 1.0
```
The output will include three contact maps along the allosteric path, named state2.npy, state3.npy (transition state), and state4.npy, all generated with a cutoff of 1.0 nm.

# Database

Please refer to the Dataset folder for detailed information.

## Contact

Please contact Wang Qian (wqq@ustc.edu.cn) for technical support.
