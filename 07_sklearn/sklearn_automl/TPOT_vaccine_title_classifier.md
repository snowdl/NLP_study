```python
"""
-obtain a model that accurately classifies whether the word ‘vaccine’
is included in the paper title (i.e., whether the paper is related to vaccines). 
-maximize the performance (such as accuracy) of the classifier that predicts the presence of ‘vaccine’."
ref:https://epistasislab.github.io/tpot/latest/
"""
```




    '\n-obtain a model that accurately classifies whether the word ‘vaccine’\nis included in the paper title (i.e., whether the paper is related to vaccines). \n-maximize the performance (such as accuracy) of the classifier that predicts the presence of ‘vaccine’."\nref:https://epistasislab.github.io/tpot/latest/\n'




```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
```


```python
# loading data
dataset_path = '11_data/metadata.csv'
metadata = pd.read_csv(dataset_path)
```

    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_9037/2775270343.py:3: DtypeWarning: Columns (1,4,5,6,13,14,15,16) have mixed types. Specify dtype option on import or set low_memory=False.
      metadata = pd.read_csv(dataset_path)



```python
# Create label
metadata['label'] = metadata['title'].str.contains('vaccine', case=False, na=False).astype(int)
```


```python
# Extract titles and labels, fill missing titles with empty string
texts = metadata['title'].fillna('')
labels = metadata['label']
```


```python
# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
```


```python
# 5. TF-IDF
vectorizer = TfidfVectorizer(max_features=1500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```


```python
# Create TPOTClassifier with limited parallel jobs to avoid distributed worker error
# generations: number of iterations to evolve pipelines
# population_size: number of pipelines in each generation
# random_state: for reproducibility
# n_jobs=1: use single core to prevent "No valid workers found" error from dask
tpot = TPOTClassifier(generations=3, population_size=15, random_state=42, n_jobs=1, max_time_mins=30)

# Fit TPOT model on training data
tpot.fit(X_train_vec, y_train)

# Export the best pipeline found to a Python script file
tpot.export('tpot_best_pipeline.py')
```

    /Users/jessicahong/.pyenv/versions/nlp_automl_env/lib/python3.10/site-packages/tpot/tpot_estimator/estimator.py:458: UserWarning: Both generations and max_time_mins are set. TPOT will terminate when the first condition is met.
      warnings.warn("Both generations and max_time_mins are set. TPOT will terminate when the first condition is met.")
    /Users/jessicahong/.pyenv/versions/nlp_automl_env/lib/python3.10/site-packages/distributed/node.py:187: UserWarning: Port 8787 is already in use.
    Perhaps you already have a cluster running?
    Hosting the HTTP server on port 53147 instead
      warnings.warn(
    Generation:   0%|                                         | 0/5 [47:26<?, ?it/s]
    Generation:   0%|                                         | 0/5 [43:06<?, ?it/s]
    Generation:   0%|                                         | 0/3 [00:00<?, ?it/s]/Users/jessicahong/.pyenv/versions/nlp_automl_env/lib/python3.10/site-packages/stopit/__init__.py:10: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      import pkg_resources



```python

```
