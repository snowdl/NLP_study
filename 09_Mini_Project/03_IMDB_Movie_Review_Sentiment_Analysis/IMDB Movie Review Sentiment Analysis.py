#!/usr/bin/env python
# coding: utf-8

# In[77]:


#import libraries
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # for data visualization
from tensorflow.keras.datasets import imdb  # IMDB dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences  # for sequence padding
from tensorflow.keras.models import Sequential  # model architecture
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout  # neural network layers


# In[37]:


# 1. Load the IMDb dataset (top 10,000 most frequent words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)


# In[38]:


#2. Print basic dataset information
print('Number of training reviews:', len(X_train))
print('Number of test reviews:', len(X_test))
num_classes = len(set(y_train))
print('Number of categories:', num_classes)


# In[78]:


# 3. Print frequency of each label (class distribution)
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("Frequency of each label:")
print(np.asarray((unique_elements, counts_elements)))

# 0 = negative, 1 = positive, [12500 12500]] = num of samples


# In[40]:


# 4. Show first training review and its length
print('First training review:', X_train[0])
print('Review length:', len(X_train[0]))


# In[41]:


# 5. Analyze review length distribution
reviews_length = [len(review) for review in X_train]
print('Maximum review length:', np.max(reviews_length))
print('Average review length:', np.mean(reviews_length))


# In[42]:


plt.subplot(1, 2, 1)
plt.boxplot(reviews_length)
plt.subplot(1, 2, 2)
plt.hist(reviews_length, bins=50)
plt.show()


# In[79]:


# 6. Pad sequences to fixed length (max_len=100)
#Sequences shorter than 100 are padded with zeros at the beginning; longer sequences are truncated to 100.
max_len = 100
X_train_padded = pad_sequences(X_train, maxlen=max_len)
X_test_padded = pad_sequences(X_test, maxlen=max_len)


# In[81]:


# 7. Prepare index-to-word dictionary to decode reviews
#Get the word-to-index mapping from the IMDb dataset and create an index-to-word dictionary.
#add 3 to each original index because the first three indices are reserved for special tokens (<pad>, <sos>, <unk>).

word_to_index = imdb.get_word_index()
index_to_word = {value + 3: key for key, value in word_to_index.items()}
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token


# In[45]:


# Print decoded first training review (for verification)
print('Decoded first training review:')
print(' '.join([index_to_word.get(index, '?') for index in X_train[0]]))


# In[46]:


# Print decoded first training review (for verification)
print('Decoded first training review:')
print(' '.join([index_to_word.get(index, '?') for index in X_train[0]]))


# In[48]:


# 8. Define the model
vocab_size = 10000  # Number of unique words considered


# In[52]:


# Basic LSTM model
"""
model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
"""


# In[82]:


#Bidirectional LSTM with Dropout (comment out Option A if using this)
# Define the model (no input_length)
model = Sequential([
    Embedding(vocab_size, 128), # converts word indices to 128-dimensional dense vectors
    Bidirectional(LSTM(64)), #bidirectional LSTM layer that processes requences forward and backword with 64 units
    Dropout(0.5), #applies 50% of dropout to reduce overfitting
    Dense(1, activation='sigmoid')
])


# In[86]:


#Build the model manually (specify input shape)
model.build(input_shape=(None, 100))  # (batch_size, sequence_length)


# In[88]:


#Uses binary_crossentropy as the loss function for binary classification
#adam optimizer for efficient and popular gradient-based optimization.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[89]:


# Show model summary
model.summary()


# In[90]:


# 9. Train the model
#Run 5 epochs with a batch size of 64 samples per iteration.
#20% of the training data for validation 
model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)


# In[22]:


#evaluation 


# In[55]:


# 1. Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Accuracy: {accuracy:.4f}')


# In[56]:


# 2. Generate predicted probabilities
y_pred = model.predict(X_test_padded)


# In[57]:


# 3. Convert probabilities to binary class labels (threshold = 0.5)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()


# In[58]:


print("Predicted labels (first 10):", y_pred_labels[:10])


# In[59]:


# 4. Print classification metrics (precision, recall, f1-score)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# In[60]:


print("Classification Report:")
print(classification_report(y_test, y_pred_labels))


# In[61]:


# 5. Compute and print ROC AUC score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))


# In[62]:


# 6. Plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




