# Protein Classification Comparison


This Python script leverages popular libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn to perform text classification. The code begins by importing essential modules and tools for data manipulation, visualization, and machine learning. Key functionalities include importing datasets using pandas from specified file paths and loading them into DataFrame objects (df_seq and df_char). The loaded datasets are expected to contain the necessary information for text classification tasks. The script then confirms the successful loading of datasets with a print statement. Users can extend the code to implement text preprocessing, feature extraction, and machine learning models for accurate text classification. Additionally, the script can be customized by updating the file paths in the dataset import statements to match the user's specific data location.

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Import Datasets
df_seq = pd.read_csv('../') #import from dataset
df_char = pd.read_csv('../') #import from dataset

print('Datasets have been loaded...')
```

Now we focuses on filtering the dataset to include only proteins, differentiating between character-level (protein_char) and sequence-level (protein_seq) data. The filtering is based on the 'macromoleculeType' column, ensuring only entries labeled as 'Protein' are retained. Subsequently, relevant variables are selected to create refined DataFrames (protein_char and protein_seq) containing essential information for further analysis. Specifically, 'structureId' and 'classification' are retained in protein_char, while 'structureId' and 'sequence' are retained in protein_seq. 

```python
# Filter for only proteins
protein_char = df_char[df_char.macromoleculeType == 'Protein']
protein_seq = df_seq[df_seq.macromoleculeType == 'Protein']

# Select only necessary variables to join
protein_char = protein_char[['structureId','classification']]
protein_seq = protein_seq[['structureId','sequence']]
protein_seq.head()
```

![image](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/f4e90fec-1982-4c17-92bd-e1008725b221)

```python 
protein_char.head()
```
![image](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/33928a23-9bcd-487b-a458-e764f9929518)

```python
# Join two datasets on structureId
model_f = protein_char.set_index('structureId').join(protein_seq.set_index('structureId'))
model_f.head()
```

![image](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/489f0ebc-77e6-4cde-8db3-ec85d49300c3)

```python
print('%d is the number of rows in the joined dataset' %model_f.shape[0])
```
- 346325 is the number of rows in the joined dataset

The two dataframes have officially been joined into one with 346,325 proteins. The data processing is not finished as it's important to take a look at the misingness associated with the columns.

```python
# Check NA counts
model_f.isnull().sum()
```
```
classification    1
sequence          3
dtype: int64
```

- With 346,325 proteins, it appears that simply removing missing values is acceptable.

```python
# Drop rows with missing values
model_f = model_f.dropna()
print('%d is the number of proteins that have a classification and sequence' %model_f.shape[0])
```

- 346321 is the number of proteins that have a classification and sequence

Finally, it is crucial to examine the various categories of family groups that can exist.







