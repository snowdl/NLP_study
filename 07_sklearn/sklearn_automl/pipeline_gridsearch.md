```python
import json
import os
import pandas as pd
```


```python
dataset_path = '11_data/metadata.csv'
metadata = pd.read_csv(dataset_path)
```

    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_4752/882283694.py:2: DtypeWarning: Columns (1,4,5,6,13,14,15,16) have mixed types. Specify dtype option on import or set low_memory=False.
      metadata = pd.read_csv(dataset_path)



```python
metadata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cord_uid</th>
      <th>sha</th>
      <th>source_x</th>
      <th>title</th>
      <th>doi</th>
      <th>pmcid</th>
      <th>pubmed_id</th>
      <th>license</th>
      <th>abstract</th>
      <th>publish_time</th>
      <th>authors</th>
      <th>journal</th>
      <th>mag_id</th>
      <th>who_covidence_id</th>
      <th>arxiv_id</th>
      <th>pdf_json_files</th>
      <th>pmc_json_files</th>
      <th>url</th>
      <th>s2_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ug7v899j</td>
      <td>d1aafb70c066a2068b02786f8929fd9c900897fb</td>
      <td>PMC</td>
      <td>Clinical features of culture-proven Mycoplasma...</td>
      <td>10.1186/1471-2334-1-6</td>
      <td>PMC35282</td>
      <td>11472636</td>
      <td>no-cc</td>
      <td>OBJECTIVE: This retrospective chart review des...</td>
      <td>2001-07-04</td>
      <td>Madani, Tariq A; Al-Ghamdi, Aisha A</td>
      <td>BMC Infect Dis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>document_parses/pdf_json/d1aafb70c066a2068b027...</td>
      <td>document_parses/pmc_json/PMC35282.xml.json</td>
      <td>https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02tnwd4m</td>
      <td>6b0567729c2143a66d737eb0a2f63f2dce2e5a7d</td>
      <td>PMC</td>
      <td>Nitric oxide: a pro-inflammatory mediator in l...</td>
      <td>10.1186/rr14</td>
      <td>PMC59543</td>
      <td>11667967</td>
      <td>no-cc</td>
      <td>Inflammatory diseases of the respiratory tract...</td>
      <td>2000-08-15</td>
      <td>Vliet, Albert van der; Eiserich, Jason P; Cros...</td>
      <td>Respir Res</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>document_parses/pdf_json/6b0567729c2143a66d737...</td>
      <td>document_parses/pmc_json/PMC59543.xml.json</td>
      <td>https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ejv2xln0</td>
      <td>06ced00a5fc04215949aa72528f2eeaae1d58927</td>
      <td>PMC</td>
      <td>Surfactant protein-D and pulmonary host defense</td>
      <td>10.1186/rr19</td>
      <td>PMC59549</td>
      <td>11667972</td>
      <td>no-cc</td>
      <td>Surfactant protein-D (SP-D) participates in th...</td>
      <td>2000-08-25</td>
      <td>Crouch, Erika C</td>
      <td>Respir Res</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>document_parses/pdf_json/06ced00a5fc04215949aa...</td>
      <td>document_parses/pmc_json/PMC59549.xml.json</td>
      <td>https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2b73a28n</td>
      <td>348055649b6b8cf2b9a376498df9bf41f7123605</td>
      <td>PMC</td>
      <td>Role of endothelin-1 in lung disease</td>
      <td>10.1186/rr44</td>
      <td>PMC59574</td>
      <td>11686871</td>
      <td>no-cc</td>
      <td>Endothelin-1 (ET-1) is a 21 amino acid peptide...</td>
      <td>2001-02-22</td>
      <td>Fagan, Karen A; McMurtry, Ivan F; Rodman, David M</td>
      <td>Respir Res</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>document_parses/pdf_json/348055649b6b8cf2b9a37...</td>
      <td>document_parses/pmc_json/PMC59574.xml.json</td>
      <td>https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9785vg6d</td>
      <td>5f48792a5fa08bed9f56016f4981ae2ca6031b32</td>
      <td>PMC</td>
      <td>Gene expression in epithelial cells in respons...</td>
      <td>10.1186/rr61</td>
      <td>PMC59580</td>
      <td>11686888</td>
      <td>no-cc</td>
      <td>Respiratory syncytial virus (RSV) and pneumoni...</td>
      <td>2001-05-11</td>
      <td>Domachowske, Joseph B; Bonville, Cynthia A; Ro...</td>
      <td>Respir Res</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>document_parses/pdf_json/5f48792a5fa08bed9f560...</td>
      <td>document_parses/pmc_json/PMC59580.xml.json</td>
      <td>https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
metadata.columns
```




    Index(['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id',
           'license', 'abstract', 'publish_time', 'authors', 'journal', 'mag_id',
           'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files',
           'url', 's2_id'],
          dtype='object')




```python
#binary classfication => classify jounals based on vaccine 
```


```python
"""
Workflow
레이블 만들기: 제목(title)에 'vaccine' 단어가 있으면 1, 없으면 0

데이터 전처리: 텍스트(제목)만 사용

훈련/테스트 분리

TF-IDF 벡터화

로지스틱 회귀 모델 훈련 및 평가
"""
```




    "\nWorkflow\n레이블 만들기: 제목(title)에 'vaccine' 단어가 있으면 1, 없으면 0\n\n데이터 전처리: 텍스트(제목)만 사용\n\n훈련/테스트 분리\n\nTF-IDF 벡터화\n\n로지스틱 회귀 모델 훈련 및 평가\n"




```python
#label creation
```


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
#Create binary label: 1 if 'vaccine' is in the title (case-insensitive), else 0
metadata['label'] = metadata['title'].str.contains('vaccine', case=False, na=False).astype(int)
```


```python
#Extract titles and labels, fill missing titles with empty string
texts = metadata['title'].fillna('')
labels = metadata['label']
```


```python
print(metadata['label'].value_counts())
```

    label
    0    1026555
    1      30105
    Name: count, dtype: int64



```python
print(metadata[metadata['label'] == 1]['title'].head(5))
```

    103    Rapid Identification of Malaria Vaccine Candid...
    106    DNA Vaccines against Protozoan Parasites: Adva...
    120    Antibody-Based HIV-1 Vaccines: Recent Developm...
    154    Expression of Foot-and-Mouth Disease Virus Cap...
    190    Nasal Delivery of an Adenovirus-Based Vaccine ...
    Name: title, dtype: object



```python
print(metadata[metadata['label'] == 0]['title'].head(5))
```

    0    Clinical features of culture-proven Mycoplasma...
    1    Nitric oxide: a pro-inflammatory mediator in l...
    2      Surfactant protein-D and pulmonary host defense
    3                 Role of endothelin-1 in lung disease
    4    Gene expression in epithelial cells in respons...
    Name: title, dtype: object



```python
#Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
```


```python
#Vectorize text data using TF-IDF, limit features to 1500 for efficiency
vectorizer = TfidfVectorizer(max_features=1500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```


```python
# Train Logistic Regression model with max 100 iterations
model = LogisticRegression(max_iter=100)
model.fit(X_train_vec, y_train)
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
#Predict on test data and calculate accuracy
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
```


```python
print(f"Test set accuracy: {acc:.3f}")
```

    Test set accuracy: 1.000



```python
#Cross-Validation, CV
```


```python
from sklearn.model_selection import cross_val_score
import numpy as np  
```


```python
model = LogisticRegression(max_iter=100)

# Perform 5-fold cross-validation on training data
scores = cross_val_score(model, X_train_vec, y_train, cv=5)  # 5-fold CV
```


```python
print("CV scores:", scores)
print("CV average accuracy:", np.mean(scores))
```

    CV scores: [0.99981072 0.99977524 0.99974566 0.99979298 0.99976932]
    CV average accuracy: 0.9997787840988709



```python
#GridSearchCV
"""
Hyperparameters are the settings that control the behavior or structure of a machine learning model, such as learning rate, regularization strength, or tree depth. They are not the data itself, but parameters that influence how the model learns from the data.
GridSearchCV is a tool that tries many combinations of hyperparameters and selects the combination that gives the best performance on the given dataset.
"""
```




    '\nHyperparameters are the settings that control the behavior or structure of a machine learning model, such as learning rate, regularization strength, or tree depth. They are not the data itself, but parameters that influence how the model learns from the data.\nGridSearchCV is a tool that tries many combinations of hyperparameters and selects the combination that gives the best performance on the given dataset.\n'




```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
```


```python
model = LogisticRegression(max_iter=100)
```


```python
#'C' controls the inverse of regularization strength in Logistic RegressionLower C values mean stronger regularization to prevent overfitting.
#'solver' specifies the algorithm used to optimize the model’s weights.
```


```python
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # List of values for regularization strength parameter 'C'. Smaller values mean stronger regularization.
    'solver': ['liblinear', 'lbfgs']  # List of optimization algorithms to choose from for training the Logistic Regression model.
}
```


```python
# Create a GridSearchCV object with 3-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=3)
```


```python
grid_search.fit(X_train_vec, y_train)
```

    /Users/jessicahong/.pyenv/versions/nlp_automl_env/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3, estimator=LogisticRegression(),
             param_grid={&#x27;C&#x27;: [0.01, 0.1, 1, 10],
                         &#x27;solver&#x27;: [&#x27;liblinear&#x27;, &#x27;lbfgs&#x27;]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3, estimator=LogisticRegression(),
             param_grid={&#x27;C&#x27;: [0.01, 0.1, 1, 10],
                         &#x27;solver&#x27;: [&#x27;liblinear&#x27;, &#x27;lbfgs&#x27;]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div>




```python
print("Best parameters:", grid_search.best_params_)
```

    Best parameters: {'C': 10, 'solver': 'lbfgs'}



```python
#test data= use them to compare the model's predictions with the actual values to objectively
y_pred = grid_search.predict(X_test_vec)
print("Test set accuracy:", accuracy_score(y_test, y_pred))
```

    Test set accuracy: 0.9997870649026177



```python
 #RandomizedSearchCV
```


```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
```


```python
param_dist = {
    'C': scipy.stats.loguniform(0.001, 100),
    'solver': ['liblinear', 'lbfgs']
}

```


```python
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
random_search.fit(X_train_vec, y_train)
```

    /Users/jessicahong/.pyenv/versions/nlp_automl_env/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    /Users/jessicahong/.pyenv/versions/nlp_automl_env/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=3, estimator=LogisticRegression(),
                   param_distributions={&#x27;C&#x27;: &lt;scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x3a858f610&gt;,
                                        &#x27;solver&#x27;: [&#x27;liblinear&#x27;, &#x27;lbfgs&#x27;]},
                   random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">RandomizedSearchCV</label><div class="sk-toggleable__content"><pre>RandomizedSearchCV(cv=3, estimator=LogisticRegression(),
                   param_distributions={&#x27;C&#x27;: &lt;scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x3a858f610&gt;,
                                        &#x27;solver&#x27;: [&#x27;liblinear&#x27;, &#x27;lbfgs&#x27;]},
                   random_state=42)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div>




```python
print("Best parameters:", random_search.best_params_)
```

    Best parameters: {'C': 14.528246637516036, 'solver': 'lbfgs'}



```python

```
