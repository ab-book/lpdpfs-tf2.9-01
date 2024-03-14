
#pip install xlrd 要安装

import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)


# In[2]:


all_df = pd.read_excel("titanic3.xls")
cols=['survived','name','pclass' ,'sex', 'age', 'sibsp',
      'parch', 'fare', 'embarked']
all_df=all_df[cols]


# In[3]:


msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk];test_df = all_df[~msk]


# In[4]:


def PreprocessData(raw_df):
    df=raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])
    ndarray = x_OneHot_df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    return scaledFeatures,Label


# In[5]:


train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)


# # 3. Create Model 

# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=80, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=60, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))
print(model.summary())


# # 4. Train model

# In[7]:


model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# In[8]:


train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=30, 
                         batch_size=30,verbose=2)


# # 6. Print History

# In[9]:




scores = model.evaluate(x=test_Features, 
                        y=test_Label)
print(scores[1])


# # 预测数据

# # 加入Jack & Rose数据

# In[13]:


#加入Jack & Rose数据,进行预测
Jack = pd.Series([0 ,'Jack',3, 'male'  , 23, 1, 0,  5.0000,'S'])
Rose = pd.Series([1 ,'Rose',1, 'female', 20, 1, 0, 100.0000,'S'])


# In[14]:


JR_df = pd.DataFrame([list(Jack),list(Rose)],  
                  columns=['survived', 'name','pclass', 'sex', 
                   'age', 'sibsp','parch', 'fare','embarked'])


# In[15]:


all_df=pd.concat([all_df,JR_df])


# In[16]:


all_df[-2:]


# # 进行预测

# In[17]:


all_Features,Label=PreprocessData(all_df)


# In[18]:


all_probability=model.predict(all_Features)


# In[19]:


all_probability[:10]


# In[20]:


pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
#查看Jack & Rose数据的生存几率
pd[-2:]


# In[21]:


#查看生存几率高，却没有存活
print(pd[(pd['survived']==0) &  (pd['probability']>0.9) ])


# In[22]:


try:
    model.save('titanic_mlp_model.h5')
    print('模型保存成功！，以后可以直接载入模型，不用再定义网络和编译模型！')
except:
    print('模型保存失败！')


# In[ ]:




