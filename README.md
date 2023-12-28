# protein_classification_comparison

## Dependencies:
- numpy
- pandas
- matplotlib
- textwrap
- scikit-learn
- IPython
- scipy

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from textwrap import wrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython.display import Image, clear_output
import scipy.sparse
import math
np.random.seed(7)

# Load sequence data for each protein
all_seqs_df = pd.read_csv('../../') #path from the dataset
# Load characteristic data for each protein
all_charcs_df = pd.read_csv('../../') #path from the dataset
```

## Library Imports:

- numpy, pandas, matplotlib: Fundamental libraries for data manipulation and visualization.
- textwrap: Used for wrapping text for better display.
- scikit-learn: Essential for machine learning tasks.
- scipy: Used for sparse matrix handling.

##Data Loading:

- all_seqs_df: Loads protein sequence data from '../input/protein-data-set/pdb_data_seq.csv'.
- all_charcs_df: Loads characteristic data from '../input/protein-data-set/pdb_data_no_dups.csv'.

## Random Seed:

- Sets a random seed to ensure reproducibility of results.

```python
protein_charcs = all_charcs_df[all_charcs_df.macromoleculeType == 'Protein'].reset_index(drop=True)
protein_seqs = all_seqs_df[all_seqs_df.macromoleculeType == 'Protein'].reset_index(drop=True)

print(protein_charcs.head())
# print(protein_seqs.head())
# protein_df.isna().sum()
# protein_df.columns
```


![image](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/6ec14556-0f67-4532-b7ac-890b07aff054)

Now we focuses on combining protein characteristics and sequence data. The code snippet provided performs several essential data manipulations, including filtering columns, merging dataframes, handling missing values, and filtering out proteins with an unknown function.


```python
protein_charcs = protein_charcs[['structureId','classification', 'residueCount', 'structureMolecularWeight',\
                         'crystallizationTempK', 'densityMatthews', 'densityPercentSol','phValue']]
protein_seqs = protein_seqs[['structureId','sequence']]

# combine protein characteristics df with their sequences using structureId
protein_all = protein_charcs.set_index('structureId').join(protein_seqs.set_index('structureId'))
protein_all = protein_all.dropna()

# capitalize all classification values to avoid missing any values in the next step
protein_all.classification = protein_all.classification.str.upper()

# drop all proteins with an unknown function; note -- the tilde returns the inverse of a filter
protein_all = protein_all[~protein_all.classification.str.contains("UNKNOWN FUNCTION")]

print(protein_all.head())
```

![image](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/ce46072b-9f62-4a5c-b772-38eb24cd91ae)

```python
class_count = protein_all.classification.value_counts()
functions = np.asarray(class_count[(class_count > 800)].index)
data = protein_all[protein_all.classification.isin(functions)]
data = data.drop_duplicates(subset=["classification","sequence"])  # leaving more rows results in duplciates / index related?
data.head()
```

