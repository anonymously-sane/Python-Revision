```python
#Importing libraries with their aliases
import pandas as pd
import numpy as np
import math as ma
import seaborn as sns
#import category_encoders as ce
```


```python
#Reading data
df = pd.read_csv("C:/Users/sangam.kushwaha/Desktop/Portable Python/Portable Python/notebooks/Dataset/train.csv",na_values = ['-','.',''])
```


```python
#Information on data
df.shape
df.info()
df.describe()
```


```python
# #Data Cleaning
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df['Age'] = df['Age'].apply(np.ceil)
# df['Cabin'] = df['Cabin'].fillna('NULL')
# df['Fare'] = round(df['Fare'],2)
# df['Embarked'] = df['Embarked'].replace({ 'S' : 'Southampton', 'C' :'Cherbourg' , 'Q' : 'Queensland' })
```


```python
#UNIVARIATE ANALYSIS
#Survivors
#df['Survived'].value_counts().plot.bar()
#Ticket Class
#df['Pclass'].value_counts().sort_index().plot.bar()
#Sex
#df['Sex'].value_counts().plot.bar()
#Age {removed age 0 & 1 for representation}
#df.loc[df['Age'].between(2,df['Age'].max())].Age.value_counts().sort_index().plot.line()
#Siblings & Spouses
#df['SibSp'].value_counts().sort_index().plot.bar()
#Parents & Children
#df['Parch'].value_counts().sort_index().plot.bar()
#Fare
#df['Fare'].value_counts().sort_index().plot.line()
#Port of Embarkation
#df['Embarked'].value_counts().plot.bar()
```


```python
#UNIVRIATE ANALYSIS WITH CONDITION {Survivor}
#sns.countplot(x ='Pclass',hue = 'Survived',data = df)
#sns.countplot(x ='Sex',hue = 'Survived',data = df)
#sns.displot(x = 'Age',hue = 'Survived',data = df)
#sns.countplot(x = 'SibSp',hue = 'Survived',data = df)
#sns.countplot(x = 'Parch',hue = 'Survived',data = df)
#sns.displot(x = 'Fare',hue = 'Survived',data = df)
#sns.countplot(x= 'Embarked', hue = 'Survived', data=df)
```


```python
# #BIVARIATE ANALYSIS
# temp = df[["Survived","Pclass","Age","Fare"]]
# sns.pairplot(temp, hue="Survived")
```


```python
# #CORRELATION ANALSIS
# #Computing correlation matrix
# corr = df.corr()
# #Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))
# #Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
```


```python
# BOXPLOTS to check data spread and behavior
#sns.boxplot(x="Survived", y="Age", data=df,whis=[0, 100], width=.6, palette="vlag")
#sns.boxplot(x="Pclass", y="Age", data=df,whis=[0, 100], width=.6, palette="vlag")
#sns.boxplot(x="Sex", y="Age", data=df,whis=[0, 100], width=.6, palette="vlag")
#sns.boxplot(x="Survived", y="Fare", data=df,whis=[0, 100], width=.6, palette="vlag")
#sns.boxplot(x="Pclass", y="Fare", data=df,whis=[0, 100], width=.6, palette="vlag")
#sns.boxplot(x="Sex", y="Fare", data=df,whis=[0, 100], width=.6, palette="vlag")
```


```python
# FEATURE SCALING
# 1)Gradient Descent Based Algorithms [linear regression, logistic regression, neural network] : Having features on a similar scale can help the gradient descent converge more quickly towards the minima.
# 2)Distance Based Algorithms [KNN, K-means, SVM] : We scale our data before employing a distance based algorithm so that all the features contribute equally to the result.
# 3)Tree Based Algorithms [Decision Trees] : No effect

# TECHNIQUES
# 1)Normalization : is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.
#     X=(x-xmin)/(xmax-xmin)
#   Used when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.
# 2)Standardization : is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.
#     X=(x-mean)/s.d.
#   Helpful in cases where the data follows a Gaussian distribution. However, this does not have to be necessarily true.

# NORMALIZATION SYNTAX IN PYTHON
#     from sklearn.preprocessing import MinMaxScaler
#     norm = MinMaxScaler().fit(df)
#     transformed_df = norm.transform(df)
    
# STANDARDIZATION SYNTAX IN PYTHON {apply only to mumerical columns}
#     from sklearn.preprocessing import StandardScaler
#     num_cols = ['colA','colB','colC','colD']
#     for i in num_cols:
#         scale = StandardScaler().fit(df[[i]])
#         df[i] = scale.transform(df[[i]])
```


```python
# DEALING WITH CATEGORICAL DATA
# Need to convert cateogrical data into numerical data

# TECHNIQUES
# 1)Replacing Values : Replace the respective category with desired numbers.
#     Method 1 -> labels = df['col_name'].astype('category').cat.categories.tolist()
#                 replacement_mapping = {'col_name' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#                 df.replace(replacement_mapping,inplace=TRUE)
#                 print(df['col_name'].dtypes)
# 2)Encoding Labels : Numerical labels between 0 to disctinct(categories in that column)-1.
#     Method 1 -> df['col_name'] = df['col_name'].cat.codes
#     Method 2 -> df['col_name'] = np.where(df['col_name'].str.contains('label_name'),1,0)
#     Method 3 -> from sklearn.preprocessing import LabelEncoder
#                 make_labels = LabelEncoder()
#                 df['col_name'] = make_labels.fit_transform(df['col_name'])
# 3)One Hot Encoding : Convert each category into a new column and assign a logical value to the column based on occurence.Not good if there are many categories.
#     Method 1 -> pd.get_dummies(df, columns=['col_name']), prefix=['encoded_']
#     Method 2 -> from sklearn.preprocessing import LabelBinarizer
#                 make_labels = LabelBinarizer()
#                 newdf = pd.DataFrame(make_labels.fit_transform(df['col_name']), columns=make_labels.classes_)
#                 finaldf = pd.concat([df,newdf],axis=1)
# 4)Binary Encoding : First the categories are encoded as ordinal, then those integers are converted into binary code, then the digits from that binary string are split into separate columns. This encodes the data in fewer dimensions than one-hot.
#     Method 1 -> import category_encoders as ce
#                 encoder_call = ce.BinaryEncoder(cols=['col_name'])
#                 newdf = encoder_call.fit_transform(df)  
            
# #For imported dataset we do not have any categorical data column so no encoding is required.
```


```python
# DEALING WITH DATA IMBALANCE
# Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce errors.
# Data imbalance fails to capture the minority class which is most often the point of creating the model in the first place.
# Examples : Fraud Detection, Spam Filtering, Disease Screening, etc.

# TECHNIQUES
# 1)Resampling :  Remove few samples from majority class and add few samples in minority class.
#         {Python syntax}
#         #{UNDERsampling Python syntax} 
#         a_count,b_count = df['Survived'].value_counts()
#         a = df[df['Survived']==0]
#         b = df[df['Survived']==1]
#         new_a_with_same_b_count = a.sample(b_count)
#         newdf = pd.concat([new_a_with_same_b_count,b],axis=0)
#         #{OVERsampling Python syntax}
#         new_b_with_same_a_count = b.sample(a_count, replace=TRUE)
#         newdf = pd.concat([a,new_b_with_same_a_count],axis=0)
# 2)Using imblearn package
#         #{UNDERsampling Python syntax} 
#         import imblearn
#         form imblearn.under_sampling import RandomUnderSampler
#         temp = RandomUnderSampler(random_state=42,replacement=TRUE)
#         x_temp,y_temp = temp.fit_resample(x,y)
#         #{OVERsampling Python syntax} 
#         import imblearn
#         form imblearn.over_sampling import RandomOverSampler
#         temp = RandomOverSampler(random_state=42)
#         x_temp,y_temp = temp.fit_resample(x,y)
#         #{y_temp} is resampled dataframe
                
# Following metrics can provide better insights over accuracy for an imbalanced data:
# ->Confusion matrix
# ->Precision
# ->Recall
# ->F1 Score
# ->Area Under ROC Curve

# UNDERsampling advantages & disadvantages
# (+)Helps improve run time by reducing data
# (-)Causes loss of information
# (-)Samples get biased leading to inaccurate results
# OVERsampling advantages & disadvantages 
# (+)Retains correct information and outperforms under sampling.
# (-)Cause overfitting and poor generalization.
```


```python

```
