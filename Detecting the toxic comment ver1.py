#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{Detecting The Toxic Comments With An Artificial Intelligence Model}}$$
# $$\small \color{red}{\textbf{The CopyRight @ Phuong V. Nguyen}}$$
# 
# $$\small \textbf{}$$

# # Loading lib
# 

# In[3]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[4]:


from pickle import dump
from pickle import load
Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'
from pprint import pprint


# # Loading data

# In[5]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# # Exploratory Data Analysis (EDA)
# ## Picking a sample

# In[6]:


train.head(5)


# In[7]:


test.head(5)


# ## Data size

# In[8]:


print(Bold+'1. The train data size:'+End)
print('The number of rows: %d. The number of columns: %d'%train.shape)
print(Bold+'2. The test data size:'+End)
print('The number of rows: %d. The number of columns: %d'%test.shape)


# ## Checking the missing value

# In[9]:


print(Bold+'1. The missing values in the train data:'+End)
print(train.isnull().sum())
print(Bold+'2. The missing values in the test data:'+End)
print(test.isnull().sum())


# # Data preparation
# 

# ## Creating the word cloud
# ### Defining the function

# In[10]:


def myword_cloud(data,text,token,width,height,max_font_size, fig_size):
    """
    data: input data
    text: The name of columne presents text, 
          such as "comment_text"
    token: the target feature, such as "toxic".
    width: The width of the word cloud, such as 1600
    height: The height of the word cloud, such as 800
    max_font_size: The maximum size of words, 200
    fig_size: The size of figure, such as (15, 7)
    """
    from wordcloud import WordCloud
    group=data[data[token]==1]
    group_text=group[text]
    neg_text=pd.Series(group_text).str.cat(sep='')
    myWordCloud=WordCloud(width=width, height=height,
                          max_font_size=max_font_size).generate(neg_text)
    plt.figure(figsize=fig_size)
    plt.imshow(myWordCloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"The most common words related to {token}",
              fontsize=20,fontweight='bold')
    plt.show()


# ### Showing the word cloud

# In[11]:


for i,col in enumerate(train.iloc[:,2:-1].columns):
    myword_cloud(data=train,text='comment_text',token=col
             ,width=1600,height=300,max_font_size=100, fig_size=(15,5))


# ## The input
# ### Picking up the comment column

# In[12]:


comments_train=train['comment_text']
print(Bold+'1. The first five comments in the train data:'+End)
print(comments_train.head(5))
comments_test=test['comment_text']
print(Bold+'2. The first five comments in the test data:'+End)
print(comments_test.head(5))


# ### Creating the index for words

# embed_size = 50 # how big is each word vector
# max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
# maxlen = 100 # max number of words in a comment to use

# In[13]:


max_features=30000
# Creating the function
tokenizer = Tokenizer(num_words=max_features)
# Fitting to data
tokenizer.fit_on_texts(list(comments_train))
# Turning tokens into a list of sequences
list_tokenized_word_train = tokenizer.texts_to_sequences(list(comments_train))
list_tokenized_word_test = tokenizer.texts_to_sequences(list(comments_test))
# List the results
print(Bold+'1. The train data'+End)
print('List of the first %d comments:'%       (len(comments_train.head(2))))
print(comments_train.head(2))
print('The index of words in the first %d comments:'%       (len(list_tokenized_word_train[:2])))
print(list_tokenized_word_train[:2])
print(Bold+'2. The test data'+End)
print('List of the first %d comments:'%       (len(comments_test.head(2))))
print(comments_test.head(2))
print('The index of words in the first %d comments:'%       (len(list_tokenized_word_test[:2])))
print(list_tokenized_word_test[:2])


# ### Measuring the length of each comment

# In[14]:


totalNumWords_train = [len(one_product) for one_product in list_tokenized_word_train]
print(Bold+'1. The train data:'+End)
print('The length of the first %d comments:'%       (len(totalNumWords_train[:10])))
print(totalNumWords_train[:10])
totalNumWords_test = [len(one_product) for one_product in list_tokenized_word_test]
print(Bold+'2. The test data:'+End)
print('The length of the first %d comments:'%       (len(totalNumWords_test[:10])))
print(totalNumWords_test[:10])


# ### The distribution of the length of comments

# In[15]:


sns.set(color_codes=True)
plt.figure(figsize=(12, 3))
sns.distplot(totalNumWords_train, kde=False, bins=30, color="steelblue")
plt.title('The distribution of the length of %d comments in the train data'%len(train),
         fontsize=15, fontweight='bold')
plt.ylabel('The number of comments',fontsize=12)
plt.xlabel('The number of words in each comment',fontsize=12)
plt.autoscale(enable=True, axis='both',tight=True)
plt.show()
plt.figure(figsize=(12, 3))
sns.distplot(totalNumWords_test, kde=False, bins=30, color="steelblue")
plt.title('The distribution of the length of %d comments in the test data'%len(test),
         fontsize=15, fontweight='bold')
plt.ylabel('The number of comments',fontsize=12)
plt.xlabel('The number of words in each comment',fontsize=12)
plt.autoscale(enable=True, axis='both',tight=True)
plt.show()


# $$\textbf{Comments:}$$
# Based on two Figures above, the distribution of the length of each comment varies a lot and has the right skew. Moreover, it fairs to say that the majority of comments have a length of 250 words. On the other hand, when we train neural networks for NLP, we need sequences to be in the same size, that’s why we use padding. Indeed, the length of all comments will be synchronized with 250 words using pad as follows.
# ### Padding
# 

# In[16]:


maxlen =250
X_train=pad_sequences(list_tokenized_word_train,maxlen=maxlen)
print(Bold+'The structure of the first comment in the train data:'+End)
print(X_train[:1])
X_test=pad_sequences(list_tokenized_word_test,maxlen=maxlen)
print(Bold+'The structure of the first comment in the test data:'+End)
print(X_test[:1])


# ## The output

# In[17]:


comment_types=train.iloc[:,2:-1].columns
print(Bold+'The %d types of comments:' %len(comment_types)+End)
pprint(comment_types)


# In[18]:


Y = train[comment_types].values
print(Bold+'The values of %d dependent variables:'%len(comment_types)+End)
pprint(Y[1:5])


# # Training Model
# ## Configuring Algorithm
# 
# ![Screenshot%202020-03-05%2011.03.50.png](attachment:Screenshot%202020-03-05%2011.03.50.png)

# In[20]:


embed_size = 50 # how big is each word vector
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(5, activation="sigmoid")(x)# The number of output, here is 6 types of comments
myAI = Model(inputs=inp, outputs=x)
myAI.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(myAI.summary())


# ### Training AI

# In[ ]:


print(Bold+ Underline+'The size of the train data:'+End)
print('The Input: %d. The output: %d'%(len(X_train),len(Y)))
print(Bold+ Underline+'The training procedure of AI:'+End)

start = timer()
myTrainedAI=myAI.fit(X_train, Y, batch_size=32, 
                             epochs=2, validation_split=0.3);
#myAI=model.fit(X_ai_train, Y_ai_train, batch_size=32, epochs=10,validation_split=0.0,
 #         validation_data=(X_ai_validation,Y_ai_validation));
print(Bold+"Time %.2fs" % (timer() - start))  


# # Finalizing project

# In[11]:


y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)

