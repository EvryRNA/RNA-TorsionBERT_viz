
# Data

This folder contains the data used for the visualisation. The data are organized as follows:

- `DECOYS`: contains the decoys torsional data for the three test sets. It has the following subfolders:
  - `angles`: torsional angles from native all the decoys of the three test sets, as well as the prediction from `RNA-TorsionBERT` (in the `rna_torsionbert` subfolder).
  - `pdb`: the computed angles for each of the decoys for each RNA of the different decoy test sets. 
  - `results`: the `PCC` and `ES` between scoring functions and metrics for the different test sets. 
  - `scores`: the computed metrics and scoring functions for each decoy test sets. We also added the `MAE` in another folder with the `_MAE` suffix. 
    The different metrics and scoring functions were computed using [RNAdvisor](https://github.com/EvryRNA/rnadvisor). 
- 'NATIVE': contains the torsional data for the training, validation, pre-training and the two test sets used when we trained the model. 
  - `native`: the native torsion and pseudo-torsional angles for the pre-training, training, validation and test sets (`RNA-Puzzles` and `CASP-RNA`).
  - `rna_torsionbert`: the predicted torsional angles from `RNA-TorsionBERT` for the data.
  - `spot_rna_1d`: the predicted torsional angles from `Spot-RNA 1D` for the data.