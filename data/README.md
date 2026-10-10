# Data

The experiments use the public **Credit Card Fraud Detection** dataset of the Machine Learning Group of the
Université Libre de Bruxelles (Dal Pozzolo et al., 2015): 284,807 real card transactions made by European
cardholders in September 2013, of which 492 are fraudulent.

1. Download `creditcard.csv` from Kaggle: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Place it in this folder as `data/creditcard.csv`.

`src/data.py` verifies the file by its SHA-256 checksum
(`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`) and prints a warning if it differs.
The file is not redistributed here; please refer to the dataset page for its license.
