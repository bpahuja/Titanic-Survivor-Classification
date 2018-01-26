
# coding: utf-8

# <h1>Titanic-DecisionTree Classifier</h1>

# So we have two data sets given here and we have to predict who survived on the titanic by using the features provided in thr train data set.
# I'll start by Importing the nessecary libraries and Cleaning the data set a little bit.

# In[5]:


import pandas as pd
tdata = pd.read_csv('train.csv',header=0)
tdata


# <h6>Describing the data</h6>

# In[7]:


tdata.describe()


# Changing Gender to an integer where Males Correspond to 1 and Females to 0.

# In[8]:


r= []
for w in tdata['Sex']:
   if  w == 'male':
       r.append(1)
   else:
       r.append(0)
tdata['Gender'] = r
tdata.head()


# Describing the data again

# In[9]:


tdata.describe()


# Setting up the Data

# In[10]:


required_columns = ['Pclass','SibSp','Parch','Fare','Gender']
y = tdata['Survived']
X = tdata[required_columns]


# For Starter Purposes I'll be using Descision Tree Classification Model here.
# It is fairly simple to understand model but there are better algorithmsout there that can perform better than the descision trees.

# In[11]:


from sklearn.tree import DecisionTreeClassifier
#Generating the model
model = DecisionTreeClassifier()
model.fit(X, y)


# Now That We have Fit the Model Generated on the basis of the data. We Ready the testing set.

# In[12]:


test = pd.read_csv('test.csv',header=0)
test.head()


# Setting the Gender Column

# In[13]:


r= []
for w in test['Sex']:
   if  w == 'male':
       r.append(1)
   else:
       r.append(0)
test['Gender'] = r
test.head()


# In[14]:


test.describe()


# In[15]:


required_column = ['Pclass','SibSp','Parch','Fare','Gender']
test_X = test[required_column]


# Now that Our testing set is ready lets bring in The Correct Values of the Survived Passengers into the File. 

# In[16]:


t = pd.read_csv('gender_submission.csv')
t.head()


# In[17]:


from sklearn.preprocessing import Imputer
imp = Imputer()
test_y = t[['Survived']]
test_Ximp = imp.fit_transform(test_X)


# Now that it is done, its time to predict using our model and determine the accuracy of our model.

# In[18]:


from sklearn.metrics import accuracy_score
predictions = model.predict(test_Ximp)
print(accuracy_score(test_y, predictions))


# Well Accuracy of this classifier i created isnt good so not a valid one but for starters I think i did Ok for my first Notebook and Descision Tree Classifier.
