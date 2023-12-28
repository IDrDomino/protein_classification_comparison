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

```
# Look at classification type counts
counts = model_f.classification.value_counts()
print(counts)

#plot counts
plt.figure()
sns.distplot(counts, hist = False, color = 'purple')
plt.title('Count Distribution for Family Types')
plt.ylabel('% of records')
plt.show()
```
```
HYDROLASE                                  46336
TRANSFERASE                                36424
OXIDOREDUCTASE                             34321
IMMUNE SYSTEM                              15615
LYASE                                      11682
HYDROLASE/HYDROLASE INHIBITOR              11218
TRANSCRIPTION                               8919
VIRAL PROTEIN                               8495
TRANSPORT PROTEIN                           8371
VIRUS                                       6972
SIGNALING PROTEIN                           6469
ISOMERASE                                   6356
LIGASE                                      4964
MEMBRANE PROTEIN                            4891
PROTEIN BINDING                             4884
STRUCTURAL PROTEIN                          4335
CHAPERONE                                   4156
STRUCTURAL GENOMICS, UNKNOWN FUNCTION       3548
SUGAR BINDING PROTEIN                       3474
DNA BINDING PROTEIN                         3199
PHOTOSYNTHESIS                              3139
ELECTRON TRANSPORT                          3047
TRANSFERASE/TRANSFERASE INHIBITOR           3032
METAL BINDING PROTEIN                       3023
CELL ADHESION                               2999
UNKNOWN FUNCTION                            2842
PROTEIN TRANSPORT                           2674
TOXIN                                       2626
CELL CYCLE                                  2496
RNA BINDING PROTEIN                         1969
                                           ...  
DE NOVO PROTEIN, TOXIN                         1
INTRACELLULAR LIPID TRANSPORT PROTEIN          1
AMINE/CARBOXYLATE LIGASE                       1
De Novo Protein, hydrolase                     1
Immune System/Protein Binding                  1
Transport Protein/Membrane Protein             1
COLD-ACTIVITY                                  1
LIPOPOLYSACCHARIDE-BINDING PROTEIN             1
TRANSFERASE(THIOSULFATE,CYANIDE SULFUR)        1
LIGASE, ISOMERASE                              1
XYLANASE A                                     1
Cytokine receptor                              1
LATE PROTEIN                                   1
Odorant Binding Protein                        1
ADP-RIBOSE BINDING PROTEIN                     1
HYDROLASE ANGIOGENESIS                         1
FERREDOXIN                                     1
HYDROLASE  INHIBITOR                           1
OXIDORECCTASE                                  1
OXIDOREDUCTASE (NADP)                          1
PROHORMONE                                     1
cellulosome                                    1
cell cycle/inhibitor                           1
HYDROLAST/HYDROLASE INHIBITOR                  1
ANTIMICROBIAL PROTEIN, MEMBRANE PROTEIN        1
SIGANLING PROTEIN,TRANSFERASE                  1
RNASE-2                                        1
NITROGEN REGULATORY PROTEIN                    1
hydrolase, cell cycle                          1
Blood Coagulation,OXIDOREDUCTASE               1
Name: classification, Length: 4468, dtype: int64
```

![__results___13_1](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/f002a934-f68d-4354-b47e-12b0b23f2169)

The counts for different family types exhibit a broad distribution. Consider filtering for a specific family type based on a threshold, such as 1,000 records. This quantity appears sufficient to enable a machine learning model to discern patterns for a particular class.


```
# Get classification types where counts are over 1000
types = np.asarray(counts[(counts > 1000)].index)

# Filter dataset's records for classification types > 1000
data = model_f[model_f.classification.isin(types)]

print(types)
print('%d is the number of records in the final filtered dataset' %data.shape[0])
```
```
['HYDROLASE' 'TRANSFERASE' 'OXIDOREDUCTASE' 'IMMUNE SYSTEM' 'LYASE'
 'HYDROLASE/HYDROLASE INHIBITOR' 'TRANSCRIPTION' 'VIRAL PROTEIN'
 'TRANSPORT PROTEIN' 'VIRUS' 'SIGNALING PROTEIN' 'ISOMERASE' 'LIGASE'
 'MEMBRANE PROTEIN' 'PROTEIN BINDING' 'STRUCTURAL PROTEIN' 'CHAPERONE'
 'STRUCTURAL GENOMICS, UNKNOWN FUNCTION' 'SUGAR BINDING PROTEIN'
 'DNA BINDING PROTEIN' 'PHOTOSYNTHESIS' 'ELECTRON TRANSPORT'
 'TRANSFERASE/TRANSFERASE INHIBITOR' 'METAL BINDING PROTEIN'
 'CELL ADHESION' 'UNKNOWN FUNCTION' 'PROTEIN TRANSPORT' 'TOXIN'
 'CELL CYCLE' 'RNA BINDING PROTEIN' 'DE NOVO PROTEIN' 'HORMONE'
 'GENE REGULATION' 'OXIDOREDUCTASE/OXIDOREDUCTASE INHIBITOR' 'APOPTOSIS'
 'MOTOR PROTEIN' 'PROTEIN FIBRIL' 'METAL TRANSPORT'
 'VIRAL PROTEIN/IMMUNE SYSTEM' 'CONTRACTILE PROTEIN' 'FLUORESCENT PROTEIN'
 'TRANSLATION' 'BIOSYNTHETIC PROTEIN']
278866 is the number of records in the final filtered dataset
```

```
# Split Data
X_train, X_test,y_train,y_test = train_test_split(data['sequence'], data['classification'], test_size = 0.2, random_state = 1)

# Create a Count Vectorizer to gather the unique elements in sequence
vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))

# Fit and Transform CountVectorizer
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

#Print a few of the features
print(vect.get_feature_names()[-20:])
```
The train_test_split function is employed to partition the data into training and testing subsets (X_train, X_test, y_train, y_test), with 80% of the data designated for training and the remaining 20% for testing.

Subsequently, a Count Vectorizer (vect) is initialized with the choice of character-level analysis (analyzer = 'char_wb') and a specified n-gram range of (4,4), capturing sequences of four characters. The vectorizer is then fitted to the training data (X_train) and transforms both the training and testing datasets into numerical feature matrices (X_train_df and X_test_df, respectively).


```
['zhhh', 'ziar', 'zigi', 'ziwz', 'zkal', 'zkky', 'zknt', 'zkyh', 'zlik', 'zlzk', 'zpvm', 'zrgd', 'zrvi', 'ztvl', 'ztzk', 'zvbd', 'zvib', 'zvka', 'zwdl', 'zzvb']
```

```
# Make a prediction dictionary to store accuracys
prediction = dict()

# Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df, y_train)
NB_pred = model.predict(X_test_df)
prediction["MultinomialNB"] = accuracy_score(NB_pred, y_test)
print( prediction['MultinomialNB'])
```
```
0.7638505396779861
```

A prediction dictionary (prediction) is initialized to store accuracy scores for different models. The focus here is on implementing a Naive Bayes model for text classification using the MultinomialNB class from scikit-learn.

The Multinomial Naive Bayes model is instantiated as model, and it is trained on the training data (X_train_df and y_train). The trained model is then used to predict the classifications for the testing data (X_test_df), and the predictions are stored in the NB_pred variable. The accuracy score of the Naive Bayes model on the testing data is calculated using the accuracy_score function and stored in the prediction dictionary under the key "MultinomialNB". Finally, the accuracy score is printed to provide a quick evaluation metric for the performance of the Naive Bayes model.

```
# Adaboost
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train_df,y_train)
ADA_pred = model.predict(X_test_df)
prediction["Adaboost"] = accuracy_score(ADA_pred , y_test)
print(prediction["Adaboost"])
```

```
0.18614408147165346
```

```
# Plot confusion matrix
conf_mat = confusion_matrix(y_test, NB_pred, labels = types)

#Normalize confusion_matrix
conf_mat = conf_mat.astype('float')/ conf_mat.sum(axis=1)[:, np.newaxis]

# Plot Heat Map
fig , ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.heatmap(conf_mat)
```

![__results___22_1](https://github.com/IDrDomino/protein_classification_comparison/assets/154571800/dc6980ba-8bfe-4f8a-8ce5-10ecfeec2e2a)


