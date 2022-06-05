#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
# numpy
# for linear algebra
# 
# pandas
# for data processing & deal with CSV data
# 
# mataplotlib
# for visualization
# 
# seaborn
# for statistical data visualization

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[6]:


sms=pd.read_csv('spam.csv',encoding='latin-1')
sms.head()


# In[7]:


# leave only data to use
sms.dropna(how="any", inplace=True, axis=1)
sms.columns = ['label', 'message']
sms.head()


# # Exploratory Data Analysis (EDA)

# In[8]:


sms.describe()


# Label divided in only two unique data

# In[9]:


sms.groupby('label').describe()


# There's 4825 ham rows and 747 spam rows data.

# In[10]:


# convert label to a numerical variable
# ham to 0 spam to 1
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
sms.head()


# In[11]:


# add len data 
sms['message_len'] = sms.message.apply(len)
sms.head()


# # Visualize for Compare the length between Ham and Spam Message

# In[12]:


plt.figure(figsize=(12, 8))

sms[sms.label=='ham'].message_len.plot(bins=35, kind='hist', color='blue', 
                                       label='Ham messages', alpha=0.6)
sms[sms.label=='spam'].message_len.plot(kind='hist', color='red', 
                                       label='Spam messages', alpha=0.6)
plt.legend()
plt.xlabel("Message Length")

We could find the spam messages usaually has a long message. I think it's because almost the spam has lots of information in it.
# In[13]:


sms[sms.label=='ham'].describe()

Let's find the message which has a longest lenght!(==910)
# In[14]:


sms[sms.message_len == 910].message.iloc[0]


# # Text Pre-processing

Our main issue with our data is that it is all in text format (strings). The classification algorithms that we usally use need some sort of numerical feature vector in order to perform the classification task. There are actually many methods to convert a corpus to a vector format. The simplest is the bag-of-words approach, where each unique word in a text will be represented by one number.

In this section we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers).

As a first step, let's write a function that will split a message into its individual words and return a list. We'll also remove very common words, ('the', 'a', etc..). To do this we will take advantage of the NLTK library. It's pretty much the standard library in Python for processing text and has a lot of useful features. We'll only use some of the basic ones here.

Let's create a function that will process the string in the message column, then we can just use apply() in pandas do process all the text in the DataFrame. Removing punctuation.
# In[15]:


import string
from nltk.corpus import stopwords ## for removing puctuating

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])


# In[16]:


sms.head()

message_len seems decreased!
# # Let's Tokenize!!
# We will use python collections.counter to count the letter of the word contains

# In[17]:


sms['clean_msg'] = sms.message.apply(text_process)
sms.head()


# In[18]:


type(stopwords.words('english'))


# # from collections import Counter
# 
# words = sms[sms.label=='ham'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])
# ham_words = Counter()
# 
# for msg in words:
#     ham_words.update(msg)
#     
# print(ham_words.most_common(50))

# In[20]:


words = sms[sms.label=='spam'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])
spam_words = Counter()

for msg in words:
    spam_words.update(msg)
    
print(spam_words.most_common(50))


# # Vectorization
# Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.
# 
# Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.
# 
# We'll do that in three steps using the bag-of-words model:
# 
# Count how many times does a word occur in each message (Known as term frequency)
# Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

# In[21]:


# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.clean_msg
y = sms.label_num
print(X.shape)
print(y.shape)


# In[22]:


# split X and y into training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer

# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(X_train)


# In[24]:


# learn training data vocabulary, then use it to create a document-term matrix
X_train_dtm = vect.transform(X_train)


# In[25]:


# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)


# In[26]:


# examine the document-term matrix
X_train_dtm


# In[27]:


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# In[28]:


from sklearn.feature_extraction.text import TfidfTransformer
# Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification.
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)


# # Building and evaluating a model
# We will use multinomial Naive Bayes:
# 
# The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

# In[29]:


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[30]:


# train the model using X_train_dtm (timing it with an IPython "magic command")
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[31]:


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[32]:


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[33]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[34]:


X_test.shape


# In[35]:


# print message text for false positives (ham incorrectly classifier)
# X_test[(y_pred_class==1) & (y_test==0)]
X_test[y_pred_class > y_test]


# In[36]:


# print message text for false negatives (spam incorrectly classifier)
X_test[y_pred_class < y_test]


# In[37]:


# example of false negative 
X_test[5]


# In[38]:


# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[39]:


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# In[40]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), 
                 ('tfid', TfidfTransformer()),  
                 ('model', MultinomialNB())])
pipe.fit(X_train, y_train)


# In[41]:


y_pred = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[42]:


metrics.confusion_matrix(y_test, y_pred)


# # Comparing models
# We will compare multinomial Naive Bayes with logistic regression:
# 
# Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

# In[43]:


# import an instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')


# In[44]:


# train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')


# In[45]:


# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)


# In[46]:


# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[47]:


# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)
0.9842067480258435


# In[48]:


metrics.confusion_matrix(y_test, y_pred_class)


# In[49]:



# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# # The result of comparing

# In[50]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2)
model = ['Multinomial Naive Bayes model', 'LogisticRegression']
values = [metrics.roc_auc_score(y_test, y_pred_prob)
,metrics.roc_auc_score(y_test, y_pred_prob)]

plt.bar(x, values)
plt.xticks(x, model)

plt.show()


# Multinomial Naive Bayes model's AUC is higher than LogisticRegrssion but has almost same values.

# In[ ]:




