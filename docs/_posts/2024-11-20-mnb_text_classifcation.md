---
layout: post
title: "Multinomial Naive Bayes for Text Classification  "
date: 2024-11-20
categories: Projects
---

In  this project I implement a Multinomial Naive Bayes model to classify text data. Multinomial Naive Bayes works well for the following reasons:

- Handles high-dimensional data
- Robust to small datasets 
- Fast training and prediction 
- Works well for sparse data 
- Is an interpretable model 


### Import Statements


```python
import spacy
import numpy as np 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, auc
```

I use this <a href="http://mlg.ucd.ie/datasets/bbc.html"> dataset </a>


```python
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
```

### Note: 
Arrays are homogeneous (all elements are of the same type) while lists are heterogeneous(elements can be diifferent)

Arrays have a fixed size, whereas lists are dynamic 

Lists in python have more built in funtions 


```python
# documents come in 5 folders, put them all together into one list 
files = sorted(list(Path('bbc').glob('**/*.txt')))
doc_list = [] 

for i,file in enumerate(files):
    # get folder name
    topic = file.parts[-2]
    article = file.read_text(encoding='latin1').split('\n')
    heading = article[0].strip()
    body = ' '.join([l.strip() for l in article[1:]])
    doc_list.append([topic, heading, body])
```


```python
# create dataframe 
docs = pd.DataFrame(doc_list, columns=['topic','heading','body'])
docs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2225 entries, 0 to 2224
    Data columns (total 3 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   topic    2225 non-null   object
     1   heading  2225 non-null   object
     2   body     2225 non-null   object
    dtypes: object(3)
    memory usage: 52.3+ KB


Here is a look into the data:

```python
docs.sample(10)
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
      <th>topic</th>
      <th>heading</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1123</th>
      <td>politics</td>
      <td>Labour pig poster 'anti-Semitic'</td>
      <td>The Labour Party has been accused of anti-Sem...</td>
    </tr>
    <tr>
      <th>325</th>
      <td>business</td>
      <td>Senior Fannie Mae bosses resign</td>
      <td>The two most senior executives at US mortgage...</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>sport</td>
      <td>Campbell lifts lid on United feud</td>
      <td>Arsenal's Sol Campbell has called the rivalry...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>sport</td>
      <td>Owen determined to stay in Madrid</td>
      <td>England forward Michael Owen has told the BBC...</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>politics</td>
      <td>Voters 'don't trust politicians'</td>
      <td>Eight out of 10 voters do not trust politicia...</td>
    </tr>
    <tr>
      <th>1633</th>
      <td>sport</td>
      <td>Woodward eyes Brennan for Lions</td>
      <td>Toulouse's former Irish international Trevor ...</td>
    </tr>
    <tr>
      <th>1006</th>
      <td>politics</td>
      <td>Kilroy launches 'Veritas' party</td>
      <td>Ex-BBC chat show host and East Midlands MEP R...</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>tech</td>
      <td>Microsoft gets the blogging bug</td>
      <td>Software giant Microsoft is taking the plunge...</td>
    </tr>
    <tr>
      <th>302</th>
      <td>business</td>
      <td>Brazil plays down Varig rescue</td>
      <td>The Brazilian government has played down clai...</td>
    </tr>
    <tr>
      <th>454</th>
      <td>business</td>
      <td>Qantas considers offshore option</td>
      <td>Australian airline Qantas could transfer as m...</td>
    </tr>
  </tbody>
</table>
</div>


We are going to classify articles based on these five categories:


```python
docs.topic.value_counts(normalize=True).to_frame('count').style.format({'count': '{:,.2%}'.format})
```




<style type="text/css">
</style>
<table id="T_479be">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_479be_level0_col0" class="col_heading level0 col0" >count</th>
    </tr>
    <tr>
      <th class="index_name level0" >topic</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_479be_level0_row0" class="row_heading level0 row0" >sport</th>
      <td id="T_479be_row0_col0" class="data row0 col0" >22.97%</td>
    </tr>
    <tr>
      <th id="T_479be_level0_row1" class="row_heading level0 row1" >business</th>
      <td id="T_479be_row1_col0" class="data row1 col0" >22.92%</td>
    </tr>
    <tr>
      <th id="T_479be_level0_row2" class="row_heading level0 row2" >politics</th>
      <td id="T_479be_row2_col0" class="data row2 col0" >18.74%</td>
    </tr>
    <tr>
      <th id="T_479be_level0_row3" class="row_heading level0 row3" >tech</th>
      <td id="T_479be_row3_col0" class="data row3 col0" >18.02%</td>
    </tr>
    <tr>
      <th id="T_479be_level0_row4" class="row_heading level0 row4" >entertainment</th>
      <td id="T_479be_row4_col0" class="data row4 col0" >17.35%</td>
    </tr>
  </tbody>
</table>




The parameter  `stratify = y` ensures that when the data is split into training and testing sets, the proportion of classes is preserved. 

Without `stratify = y` there may be an unbalanced distributiion of classes between training and testing sets, especially if some classes are less frequent in the original dataset. 

With `stratify = y` the splot respected the distribution of the different classes in the dataset. 

For example, if topic 1 represents 20% of the original dataset, it will also represent approximately 20% of both the training and testing sets. 


```python
# classify news articles

# create integer class values
y = pd.factorize(docs.topic)[0]
x = docs.body 
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1,stratify=y)
```

### Vectorize Text Data

`CountVectorizer()` converts a collection of text documents into a matrix of token/word counts. First the data is tokenized by being split into individual words. Then, a vocabulary of unique tokens across the entire corpus is built. Finally, a word count matrix is created where each row corresponds to a document and each column corresponds to the count of a unique word in the vocabulary. 


```python
vectorizer = CountVectorizer(stop_words=None)
x_train_dtm = vectorizer.fit_transform(x_train)
x_test_dtm = vectorizer.transform(x_test)
```


```python
x_train_dtm.shape, x_test_dtm.shape
```




    ((1668, 25951), (557, 25951))



### Train Multi-Class Naive Bayes Model 

Naive Bayes is based on Bayes Theorem:

First equation is here \\({P(C\|X) = \frac{P(X∣C)P(C)}{P(X)}}\\) 


where C is the class and X is the feature. 


Some limitations are its **feature independence assumption** and its **zero frequency problem**. There is a strong assumption that features are condiditionally independent which does not always hold, depending on the data. Also, if a class has a zero probability for a given feature, the entire product becomes zero. 

Multinomial naive bayes handles frequency based features, making it effective for text classification when the number of times a word appears in an article is meaningful. 

My one concern while building this was that stop words would be frequent enough across all documents to influence the classification of topics. However, stop words are generally not topic-specific and do not provide much discriiminatory power between the classes. For example, the word "the" can appear in articles about both sports and politics. 

Naive Bayes is robust in this nature-- it naturall down weights stop words because of its probablistic nature. Stop words, being common across all classes, will have siimilar probabilities for all classes *((P(word\|class))*. Thus, these words have limited impact on the overall classification decision 

```python
nb = MultinomialNB()
nb.fit(x_train_dtm,y_train)
y_pred_class = nb.predict(x_test_dtm)
```


```python
score = accuracy_score(y_test, y_pred_class)
print(score)
```

    0.9712746858168761


Our model produces an accuracy score of 97% -- pretty good! 

### Create and plot confusion matrix


```python
# calculate the confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_class)

# display the confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap='Blues', xticks_rotation=45)

# customize plot
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

```


![image]({{site.url}}/assets/images/mnb_text_classifcation_files/mnb_text_classifcation_22_0.png)

  

### Create ROC plot for all classes


```python
y_pred_probs = nb.predict_proba(x_test_dtm)
# create binary labels for class 0
y_test_binary = (y_test == 0).astype(int)   
# probabilities for class 0
y_pred_probs_class0 = y_pred_probs[:, 0]     
```


```python
# get the unique class labels
classes = np.unique(y_test)  
n_classes = len(classes)

# binarize the output for multi-class ROC
y_test_binarized = label_binarize(y_test, classes=classes)

# initialize plot
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', n_classes)

# plot ROC for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})", color=colors(i))

# ddd diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")

# plot settings
plt.title("One-vs-Rest ROC Curves", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
```

    
![image]({{site.url}}/assets/images/mnb_text_classifcation_files/mnb_text_classifcation_25_1.png)

ROC curves look good!