# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:25:46 2018

Titanic
1) Decision Tree 
2) Deep-Learning

@author: MARINAKH
"""

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz

#for deep-learning:
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping 
from keras.utils import to_categorical

# Figures inline and set visualization style
%matplotlib inline
sns.set()

df_test = pd.read_csv('test.csv')
df_train= pd.read_csv('train.csv')

df_test.head()
df_train.head()

# Exploratory analysis (EDA)
df_test.shape
df_train.shape
df_test.columns
df_train.columns
df_train.info()
df_test.info()
df_train.tail()
df_train.describe()
# describe for all , incl objects:
df_train.describe(include=['O'])
df_train.describe(include='all')
df_train['Embarked'].describe()
df_train['Embarked'].head(20)

# check duplicates:
duplicates = df_train.duplicated()
duplicates.describe()
duplicates = df_test.duplicated()
duplicates.describe()


 ## EDA on Feature Variables:
sns.countplot(x='Survived', data=df_train);
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);
df_train.Survived.sum()
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())

""" Cleaning data:"""

# Concatenate both df:
df_merged = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
df_merged.describe(include='all')
df_merged.info()

# Check nulls:
df_train.isnull().sum()
df_test.isnull().sum()
df_merged.isnull().sum()
 
# Category from Sex and Embarked,
df_merged.Sex.value_counts(dropna=False)
df_merged['Sex']=df_merged['Sex'].astype('category')

df_merged.Embarked.value_counts(dropna=False)
df_merged.Embarked = df_merged.Embarked.astype('category')
df_merged.dtypes
 

# treating Nulls in age:
age_by_sex_class = df_merged.groupby(['Sex', 'Pclass'])['Age']

def impute_median(series):
    return series.fillna(series.median())

df_merged.Age = age_by_sex_class.transform(impute_median)
df_merged.tail(10)


# treating Nulls in embarked 
df_merged[df_merged['Embarked'].isnull()==True]
df_merged.groupby(['Pclass'])['Embarked'].describe()
#Embarked_mode = df_merged['Embarked'].mode()[0]
df_merged['Embarked'].fillna('S', inplace=True)

# treating Nulls in Fare 
df_merged[df_merged['Fare'].isnull()==True]
df_merged.Fare[(df_merged.Pclass==3) & (df_merged.Sex=='male') & (df_merged.Embarked=='S')].describe()
df_merged.Fare.fillna(8.05, inplace=True)

# treating  Nulls in Cabin 
df_merged['Cabin'] = df_merged.Cabin.notnull().astype(int)
df_merged.tail(10)     
df_merged.groupby(['Cabin']).describe().T
 
# Extracting Titles:
titles = set()
for name in df_merged['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

print(titles) 

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}    

# adding new column:
df_merged['Title'] = df_merged['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
df_merged['Title'] = df_merged.Title.map(Title_Dictionary)

df_merged.tail(10) 
df_merged['Title'].unique()
df_merged[df_merged['Title'].isnull()==True]
 
# most machine learning models work input features that are numerical
# use the pandas function .get_dummies() to do so:
df_merged.dtypes
df_merged_1 = pd.get_dummies(df_merged, columns=['Sex',], drop_first=True)
df_merged_1 = df_merged_1.drop(['PassengerId', 'Name', 'Ticket'],axis = 1)
df_merged_1 = pd.get_dummies(df_merged_1, columns=['Embarked', 'Title'], drop_first=True)
df_merged_1.head()
df_merged_1.info()

# split back into 2 DF
data_train = df_merged_1.iloc[:891]
data_test = df_merged_1.iloc[891:]
data_train_y = df_train['Survived']
data_train.head()

# We'll use scikit-learn, which requires your data as arrays, not DataFrames so transform them:
X_train = data_train.values
X_test = data_test.values
y_train = data_train_y.values

""" Modeling """

""" V0. Predict that everyone will die"""
def predictions_v0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data_train.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions_v0 = predictions_v0(data_train)

# Define accuracy function:
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' of manual prediction
print(accuracy_score(data_train_y, predictions_v0))


""" V1. Create simple manual model """

def predictions_manual(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
        predictions = []
    for _, passenger in data_train.iterrows():
        if passenger['Sex_male'] =='1' and passenger['Pclass'] == 3 and passenger['Age'] > 20 and  passenger['Age'] < 60:
            predictions.append(0)
        elif passenger['Sex_male'] =='0':
            predictions.append(1)
        elif passenger['Age'] <10:
            predictions.append(1)
        else:
            predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions_v1 = predictions_manual(data_train)

# Test the 'accuracy_score' of manual prediction
print(accuracy_score(data_train_y, predictions_v1)) #63%


""" V2. Create Decision tree """

## Now you get to build your decision tree classifier! 
#First create such a model with max_depth=3 and then fit it your data. 
#Note that you name your model clf, which is short for "Classifier".
# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

clf.feature_importances_
clf.n_features_
headers = data_train.columns

Dcs_Tree_Weigts = pd.DataFrame(clf.feature_importances_.T, index = eaders = headers)

 # Make predictions and store in 'Survived' column of df_test
predictions_v3 = clf.predict(X_test)
Dcs_Tree_prediction_v3 = df_test[['PassengerId']] 
Dcs_Tree_prediction_v3['Survived'] = predictions_v3
Dcs_Tree_prediction_v3.to_csv('Dcs_Tree_prediction_v3.csv', index=False)

print("Train score: " + str(float(round(clf.score(X_train, y_train), 5))) + " Test score : 0.77990")


""" HOW DO YOU TELL WHETHER YOU ARE OVERFITTING OR UNDERFITTING?
One way is to hold out a test set from your training data. 
You can then fit the model to your training data, 
make predictions on your test set and see how well your prediction does 
on the test set"""

#split your original training data into training and test sets:
X_train_66, X_train_hold_33, y_train_66, y_train_hold_33 = train_test_split(
    X_train, y_train, test_size=0.33, random_state=42, stratify=y_train)

#Now, you'll iterate over values of max_depth ranging from 1 to 9 and plot the accuracy of the models on training and test sets:
# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a k-NN Classifier with k neighbors: knn
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train_66, y_train_66)

    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train_66, y_train_66)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_train_hold_33, y_train_hold_33)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# same with min_samples_leaf=30
clf_3_6 = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=30)
clf_3_6.fit(X_train, y_train)

predictions_v6 = clf_3_6.predict(X_test)
Dcs_Tree_prediction_v6 = df_test[['PassengerId']] 
Dcs_Tree_prediction_v6['Survived'] = predictions_v6
Dcs_Tree_prediction_v6.to_csv('Dcs_Tree_prediction_v6.csv', index=False)

print("Train score: " + str(float(round(clf_3_6.score(X_train, y_train), 5))) + " Test score : 0.78947")
# score :0.78947

clf_3_6.feature_importances_

# export data of DCS tree with  names:
tree.export_graphviz(clf_3_6,out_file='tree.dot', feature_names=headers,  
                         class_names=['Survived', 'Dead'],  
                         filled=True, rounded=True)
# to visualize it: http://webgraphviz.com/

""" Neural Networks V7"""

# Save the number of columns in predictors: n_cols
n_cols = X_train.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(10, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3)
model_0_training = model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping_monitor], validation_split=0.3, verbose=False)

# Model Summary
model.summary()
model.layers
model.get_config()
model.get_weights()

#prediction: 
prediction_v7 = model.predict(X_test)
prediction_v7.max()
prediction_v7.min()
y_final = (prediction_v7 > 0.5).astype(int)

NN_v7 = df_test[['PassengerId']] 
NN_v7['Survived'] = y_final
NN_v7.to_csv('NN_v7.csv', index=False)

print("Train score: " + str(model.evaluate(X_train, y_train)) + " Test score : 0.74162")


""" V8 Network"""
# Initialising the NN
model_v8 = Sequential()

# layers
model_v8.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_shape=(n_cols,)))
model_v8.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model_v8.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model_v8.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model_v8.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model_v8.fit(X_train, y_train, callbacks=[early_stopping_monitor], validation_split=0.3, batch_size = 32, epochs = 200)

# Model Summary
model_v8.summary()
model_v8.layers
model_v8.get_config()
model_v8.get_weights()


#prediction: 
prediction_v8 = model.predict(X_test)
prediction_v8.max()
prediction_v8.min()
y_final_v8 = (prediction_v8> 0.5).astype(int)

NN_v8 = df_test[['PassengerId']] 
NN_v8['Survived'] = y_final
NN_v8.to_csv('NN_v8.csv', index=False)

print("Train score: " + str(model.evaluate(X_train, y_train)) + " Test score : 0.74162")


""" V9 Network"""
target = to_categorical(y_train)

# Initialising the NN
model_v9 = Sequential()

# layers
model_v9.add(Dense(9, activation = 'relu', input_shape=(n_cols,)))
model_v9.add(Dense(9, activation = 'relu'))
model_v9.add(Dense(5, activation = 'relu'))
model_v9.add(Dense( 2,  activation ='softmax'))

# Compiling the ANN
model_v9.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model_v9.fit(X_train, target, callbacks=[early_stopping_monitor], validation_split=0.3, batch_size = 32, epochs = 100)
