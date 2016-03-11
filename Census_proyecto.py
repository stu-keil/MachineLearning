# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:15:44 2015

@author: Stephane

Proyecto Final de Aprendizaje de Maquina


 This data was extracted from the census bureau database found at
| http://www.census.gov/ftp/pub/DES/www/welcome.html
| Donor: Ronny Kohavi and Barry Becker,
|        Data Mining and Visualization
|        Silicon Graphics.
|        e-mail: ronnyk@sgi.com for questions.
| Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
| 48842 instances, mix of continuous and discrete    (train=32561, test=16281)
| 45222 if instances with unknown values are removed (train=30162, test=15060)
| Duplicate or conflicting instances : 6
| Class probabilities for adult.all file
| Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
| Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
|
| Extraction was done by Barry Becker from the 1994 Census database.  A set of
|   reasonably clean records was extracted using the following conditions:
|   ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
|
| Prediction task is to determine whether a person makes over 50K
| a year.
|
| First cited in:
| @inproceedings{kohavi-nbtree,
|    author={Ron Kohavi},
|    title={Scaling Up the Accuracy of Naive-Bayes Classifiers: a
|           Decision-Tree Hybrid},
|    booktitle={Proceedings of the Second International Conference on
|               Knowledge Discovery and Data Mining},
|    year = 1996,
|    pages={to appear}}
|
| Error Accuracy reported as follows, after removal of unknowns from
|    train/test sets):
|    C4.5       : 84.46+-0.30
|    Naive-Bayes: 83.88+-0.30
|    NBTree     : 85.90+-0.28
|
|
| Following algorithms were later run with the following error rates,
|    all after removal of unknowns and using the original train/test split.
|    All these numbers are straight runs using MLC++ with default values.
|
|    Algorithm               Error
| -- ----------------        -----
| 1  C4.5                    15.54
| 2  C4.5-auto               14.46
| 3  C4.5 rules              14.94
| 4  Voted ID3 (0.6)         15.64
| 5  Voted ID3 (0.8)         16.47
| 6  T2                      16.84
| 7  1R                      19.54
| 8  NBTree                  14.10
| 9  CN2                     16.00
| 10 HOODG                   14.82
| 11 FSS Naive Bayes         14.05
| 12 IDTM (Decision table)   14.46
| 13 Naive-Bayes             16.12
| 14 Nearest-neighbor (1)    21.42
| 15 Nearest-neighbor (3)    20.35
| 16 OC1                     15.04
| 17 Pebls                   Crashed.  Unknown why (bounds WERE increased)
|
| Conversion of original data as follows:
| 1. Discretized agrossincome into two ranges with threshold 50,000.
| 2. Convert U.S. to US to avoid periods.
| 3. Convert Unknown to "?"
| 4. Run MLC++ GenCVFiles to generate data,test.
|
| Description of fnlwgt (final weight)
|
| The weights on the CPS files are controlled to independent estimates of the
| civilian noninstitutional population of the US.  These are prepared monthly
| for us by Population Division here at the Census Bureau.  We use 3 sets of
| controls.
|  These are:
|          1.  A single cell estimate of the population 16+ for each state.
|          2.  Controls for Hispanic Origin by age and sex.
|          3.  Controls by Race, age and sex.
|
| We use all three sets of controls in our weighting program and "rake" through
| them 6 times so that by the end we come back to all the controls we used.
|
| The term estimate refers to population totals derived from CPS by creating
| "weighted tallies" of any specified socio-economic characteristics of the
| population.
|
| People with similar demographic characteristics should have
| similar weights.  There is one important caveat to remember
| about this statement.  That is that since the CPS sample is
| actually a collection of 51 state samples, each with its own
| probability of selection, the statement only applies within
| state.


>50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import math
from scipy.stats import describe
import os

os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\Proyecto_Final")
os.listdir(".")


train_census = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",header=None,na_values=['?'],skipinitialspace=True)
test_census = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",header=None,skiprows=1,na_values=['?'],skipinitialspace=True)

train_census.head()
test_census.head()
features = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','earning_class']

train_census.columns = features
test_census.columns = features

all_census = pd.concat([train_census,test_census],ignore_index=True)
all_census.head()


all_census.loc[all_census['earning_class'] == '<=50K.', 'earning_class'] = '<=50K'
all_census.loc[all_census['earning_class'] == '>50K.', 'earning_class'] = '>50K'

''' 
Valores Faltantes
'''

d = features
miss = []
prop = []
for i in all_census.columns:
    miss.append(pd.isnull(all_census[i]).sum())
    prop.append((pd.isnull(all_census[i]).sum().astype("f64")/len(all_census)))
missing = pd.DataFrame(zip(features,miss,prop))
missing.columns = ["var","missing vals","missing proportion"]

all_census_prep = all_census.copy()
del all_census_prep["fnlwgt"]
del all_census_prep["education_num"]



all_census_prep.loc[pd.isnull(all_census_prep['workclass']),'workclass']='missing'
all_census_prep.loc[pd.isnull(all_census_prep['occupation']),'occupation']='missing'
all_census_prep.loc[pd.isnull(all_census_prep['native_country']),'native_country']='missing'


all_census_prep['workclass'].unique()
all_census_prep['workclass'].replace("State-gov","Other-Govt",True)
all_census_prep['workclass'].replace("Local-gov","Other-Govt",True)
all_census_prep['workclass'].replace("Self-emp-inc","Self-Employed",True)
all_census_prep['workclass'].replace("Self-emp-not-inc","Self-Employed",True)
all_census_prep['workclass'].replace("Without-pay","Not-Working",True)
all_census_prep['workclass'].replace("Never-worked","Not-Working",True)


all_census_prep['occupation'].replace("Adm-clerical","Admin",True)
all_census_prep['occupation'].replace("Armed-Forces","Military",True)
all_census_prep['occupation'].replace("Craft-repair","Blue-Collar",True)
all_census_prep['occupation'].replace("Exec-managerial","White-Collar",True)
all_census_prep['occupation'].replace("Farming-fishing","Blue-Collar",True)
all_census_prep['occupation'].replace("Handlers-cleaners","Blue-Collar",True)
all_census_prep['occupation'].replace("Machine-op-inspct","Blue-Collar",True)
all_census_prep['occupation'].replace("Other-service","Service",True)
all_census_prep['occupation'].replace("Priv-house-serv","Service",True)
all_census_prep['occupation'].replace("Prof-specialty","Professional",True)
all_census_prep['occupation'].replace("Protective-serv","Other-Occupations",True)
all_census_prep['occupation'].replace("Sales","Sales",True)
all_census_prep['occupation'].replace("Tech-support","Other-Occupations",True)
all_census_prep['occupation'].replace("Transport-moving","Blue-Collar",True)

all_census_prep['native_country'].replace("Cambodia","SE-Asia",True)
all_census_prep['native_country'].replace("Canada","British-Commonwealth",True)    
all_census_prep['native_country'].replace("China","China"       ,True)
all_census_prep['native_country'].replace("Columbia","South-America"    ,True)
all_census_prep['native_country'].replace("Cuba","Other"        ,True)
all_census_prep['native_country'].replace("Dominican-Republic","Latin-America",True)
all_census_prep['native_country'].replace("Ecuador","South-America"     ,True)
all_census_prep['native_country'].replace("El-Salvador","South-America" ,True)
all_census_prep['native_country'].replace("England","British-Commonwealth",True)
all_census_prep['native_country'].replace("France","Euro_1",True)
all_census_prep['native_country'].replace("Germany","Euro_1",True)
all_census_prep['native_country'].replace("Greece","Euro_2",True)
all_census_prep['native_country'].replace("Guatemala","Latin-America",True)
all_census_prep['native_country'].replace("Haiti","Latin-America",True)
all_census_prep['native_country'].replace("Holand-Netherlands","Euro_1",True)
all_census_prep['native_country'].replace("Honduras","Latin-America",True)
all_census_prep['native_country'].replace("Hong","China",True)
all_census_prep['native_country'].replace("Hungary","Euro_2",True)
all_census_prep['native_country'].replace("India","British-Commonwealth",True)
all_census_prep['native_country'].replace("Iran","Other",True)
all_census_prep['native_country'].replace("Ireland","British-Commonwealth",True)
all_census_prep['native_country'].replace("Italy","Euro_1",True)
all_census_prep['native_country'].replace("Jamaica","Latin-America",True)
all_census_prep['native_country'].replace("Japan","Other",True)
all_census_prep['native_country'].replace("Laos","SE-Asia",True)
all_census_prep['native_country'].replace("Mexico","Latin-America",True)
all_census_prep['native_country'].replace("Nicaragua","Latin-America",True)
all_census_prep['native_country'].replace("Outlying-US(Guam-USVI-etc)","Latin-America",True)
all_census_prep['native_country'].replace("Peru","South-America",True)
all_census_prep['native_country'].replace("Philippines","SE-Asia",True)
all_census_prep['native_country'].replace("Poland","Euro_2",True)
all_census_prep['native_country'].replace("Portugal","Euro_2",True)
all_census_prep['native_country'].replace("Puerto-Rico","Latin-America",True)
all_census_prep['native_country'].replace("Scotland","British-Commonwealth",True)
all_census_prep['native_country'].replace("South","Euro_2",True)
all_census_prep['native_country'].replace("Taiwan","China",True)
all_census_prep['native_country'].replace("Thailand","SE-Asia",True)
all_census_prep['native_country'].replace("Trinadad&Tobago","Latin-America",True)
all_census_prep['native_country'].replace("United-States","United-States",True)
all_census_prep['native_country'].replace("Vietnam","SE-Asia",True)
all_census_prep['native_country'].replace("Yugoslavia","Euro_2",True)

all_census_prep['relationship'].replace("Husband","Spouse",True)
all_census_prep['relationship'].replace("Wife","Spouse",True)

all_census_prep['marital_status'].replace("Married-AF-spouse","Married",True)
all_census_prep['marital_status'].replace("Married-civ-spouse","Married",True)
all_census_prep['marital_status'].replace("Married-spouse-absent","Not-Married",True)
all_census_prep['marital_status'].replace("Separated","Not-Married",True)
all_census_prep['marital_status'].replace("Divorced","Not-Married",True)

all_census_prep['education'].replace("10th","Dropout",True)
all_census_prep['education'].replace("11th","Dropout",True)
all_census_prep['education'].replace("12th","Dropout",True)
all_census_prep['education'].replace("1st-4th","Dropout",True)
all_census_prep['education'].replace("5th-6th","Dropout",True)
all_census_prep['education'].replace("7th-8th","Dropout",True)
all_census_prep['education'].replace("9th","Dropout",True)
all_census_prep['education'].replace("Assoc-acdm","Associates",True)
all_census_prep['education'].replace("Assoc-voc","Associates",True)
all_census_prep['education'].replace("Bachelors","Bachelors",True)
all_census_prep['education'].replace("Doctorate","Doctorate",True)
all_census_prep['education'].replace("HS-Grad","HS-Graduate",True)
all_census_prep['education'].replace("Masters","Masters",True)
all_census_prep['education'].replace("Preschool","Dropout",True)
all_census_prep['education'].replace("Prof-school","Prof-School",True)
all_census_prep['education'].replace("Some-college","HS-Graduate",True)


###Convierto a factores
for i in all_census_prep.columns[all_census_prep.dtypes == object]:
    all_census_prep[i] = all_census_prep[i].astype('category')
    
all_census_prep["workclass"].cat.categories

all_census_prep.ftypes


binary_data = pd.get_dummies(all_census_prep["earning_class"])
del all_census_prep["earning_class"]
all_census_prep["earning_class"] = binary_data[binary_data.columns[1]].astype("int64")
all_census_prep["earning_class"].value_counts()

fig = plt.pyplot.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(all_census_prep.shape[1]) / cols)
for i, column in enumerate(all_census_prep.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if all_census_prep.dtypes[column]!='int64':
        ###Plot histogram for categorical values
        all_census_prep[column].value_counts().plot(kind="bar", axes=ax)
    else:
        all_census_prep[column].hist(axes=ax)
        plt.pyplot.xticks(rotation="vertical")
plt.pyplot.subplots_adjust(hspace=0.7, wspace=0.2)


'''
EDA
'''
###Convierto a factores
for i in all_census.columns[all_census.dtypes == object]:
    all_census[i] = all_census[i].astype('category')

all_census.shape
all_census.columns

'''
age                  int64:dense
workclass         category:dense
fnlwgt               int64:dense
education         category:dense
education_num        int64:dense
marital_status    category:dense
occupation        category:dense
relationship      category:dense
race              category:dense
sex               category:dense
capital_gain         int64:dense
capital_loss         int64:dense
hours_per_week       int64:dense
native_country    category:dense
earning_class     category:dense
'''
all_census.ftypes
all_census.describe()

'''Reportar kurtosis y sesgo
'''
stats = describe(all_census[["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]], axis=0)
print stats.mean
print list(stats[4])

'''
count     48842
unique        5
top       White
freq      41762
Name: race, dtype: object
'''
d = []
for i in all_census.columns[all_census.dtypes == 'category']:
    d.append(all_census[i].describe())
    


###Ver todas las categorias
'''
['White' 'Black' 'Asian-Pac-Islander' 'Amer-Indian-Eskimo' 'Other']
'''
for i in all_census.columns[all_census.dtypes == 'category']:
    print all_census[i].unique()


for i in all_census.columns:
    print all_census[i].describe()
'''
<=50K    37155
>50K     11687
dtype: int64
'''
for i in all_census.columns:
    print all_census[i].value_counts()
    
    
all_census["earning_class"].value_counts()



'''
Continous Variable Analysis
'''
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(all_census['age'], bins = 10, range = (all_census['age'].min(),all_census['age'].max()))
plt.pyplot.title('Age distribution')
plt.pyplot.xlabel('Age')
plt.pyplot.ylabel('Count')
plt.pyplot.show()

all_census.hist(column="age")
all_census.boxplot(column="age")
all_census.boxplot(column="age", by = "earning_class")
all_census["age"].plot(kind="density")

'''
Cateogrical data analysis
'''
temp1 = all_census.groupby('race')["age"].count()
temp2 = all_census.groupby('race')["age"].sum()/all_census.groupby('race')["age"].count()
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Race')
ax1.set_ylabel('Count')
ax1.set_title("People by Race")
temp1.plot(kind='bar')

fig = plt.figure(figsize=(8,4))
ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Race')
ax2.set_ylabel('Age mean')
ax2.set_title("Average age per race")




all_census["race"].value_counts().plot(kind='bar')
all_census["earning_class"].value_counts().plot(kind='bar')

grouped = all_census.groupby(["earning_class","marital_status"])
grouped.size()


'''
Plot all pretty
'''
sns.pairplot(all_census_prep)

g = sns.PairGrid(all_census)
g.map(plt.scatter);

g = sns.PairGrid(all_census, hue="earning_class")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();


fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(all_census.shape[1]) / cols)
for i, column in enumerate(all_census.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if all_census.dtypes[column]!='int64':
        ###Plot histogram for categorical values
        all_census[column].value_counts().plot(kind="bar", axes=ax)
    else:
        all_census[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


'''
Private             0.694198
Self-emp-not-inc    0.079071
Local-gov           0.064207
State-gov           0.040559
Self-emp-inc        0.034704
Federal-gov         0.029319
Without-pay         0.000430
Never-worked        0.000205
dtype: float64
'''
d = []
for i in all_census.columns[all_census.dtypes == 'category']:
    d.append((all_census[i].value_counts()[0:10] / all_census.shape[0])*100)
    
    
all_census["workclass"].value_counts() / all_census.shape[0]
all_census["education"].value_counts() / all_census.shape[0]
all_census["marital_status"].value_counts() / all_census.shape[0]
all_census["occupation"].value_counts() / all_census.shape[0]
all_census["relationship"].value_counts() / all_census.shape[0]
all_census["race"].value_counts() / all_census.shape[0]
all_census["sex"].value_counts() / all_census.shape[0]
all_census["hours_per_week"].value_counts() / all_census.shape[0]
all_census["native_country"].value_counts() / all_census.shape[0]



'''
Transformar categoricas a nominales
'''
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] != 'int64':
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


# Calculate the correlation and plot it
encoded_data, encoders = number_encode_features(all_census_prep)
encoded_data.ftypes
sns.heatmap(binary_data.corr(), square=True)
plt.show()


encoders["earning_class"].classes_
encoders["sex"].classes_
encoded_data["sex"].value_counts() / all_census.shape[0]

all_census[["sex", "relationship"]].head(15)

'''
Build a classifier
'''
binary_data.corr()["earning_class"]

'''
Split and scale

'''
X_train, X_test, y_train, y_test = cross_validation.train_test_split(encoded_data[encoded_data.columns.difference(["earning_class"])], encoded_data["earning_class"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("f64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("f64"))

'''
Logistic Regression
'''

cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
X_test.shape
y_pred_prob = cls.predict_proba(X_test)###Probabilidad de pertenecer a cada una de mis clases
cls.decision_function(X_test)
y_pred = cls.predict(X_test)

np.unique(y_pred, return_counts=True)


cm = metrics.confusion_matrix(y_test, y_pred)
plt.pyplot.figure(figsize=(12,12))
plt.pyplot.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["<50K",">50K"], yticklabels=["<50K",">50K"])
plt.pyplot.ylabel("Real value")
plt.pyplot.xlabel("Predicted value")

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.pyplot.figure(figsize=(12,12))
plt.pyplot.subplot(2,1,1)
sns.heatmap(cm_normalized, annot=True, fmt="f", xticklabels=["<50K",">50K"], yticklabels=["<50K",">50K"])
plt.pyplot.ylabel("Real value")
plt.pyplot.xlabel("Predicted value")

(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))

print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)

coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort()
plt.pyplot.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()


fpr, tpr, thresholds = roc_curve(y_test,pd.DataFrame(y_pred_prob)[1])
dist=  map(math.sqrt,(1-tpr)**2+(fpr**2))
ind = dist.index(min(dist))
umbral = thresholds[ind]
y_predict_coded = []
for i in pd.DataFrame(y_pred_prob)[1]:
    if i > umbral:
        y_predict_coded.append(1)
    else:
        y_predict_coded.append(0)

cm = metrics.confusion_matrix(y_test, y_predict_coded)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["earning_class"].classes_, yticklabels=encoders["earning_class"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm_normalized, annot=True, fmt="f", xticklabels=encoders["earning_class"].classes_, yticklabels=encoders["earning_class"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")



print "F1 score: %f" % skl.metrics.f1_score(y_test, y_predict_coded)

coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort()
plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=1, label='Curva ROC')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(0,9709,1000):  
    la = round(thresholds[xy],2)
    fi = (fpr[xy],tpr[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()

'''
Using Binary Attributes
'''
census = all_census.copy()
census.columns
del census["fnlwgt"]
del census["education"]
binary_data = pd.get_dummies(census)

del binary_data["earning_class_<=50K"]

binary_data.dtypes
binary_data.columns.str.find("education_")

plt.subplots(figsize=(20,20))
sns.heatmap(binary_data.corr(), square=True)
plt.show()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns.difference(["earning_class_>50K"])], binary_data["earning_class_>50K"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)

cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["<=50k",">50K"], yticklabels=["<=50k",">50K"],square=True)
plt.ylabel("Real value")
plt.xlabel("Predicted value")




print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)

coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()
'''
Naive Bayes

'''
naive_data = all_census_prep.copy()
naive_data.head()
binary_data = pd.get_dummies(all_census_prep["earning_class"])
del naive_data["earning_class"]
naive_data["earning_class"] = binary_data[binary_data.columns[1]].astype("int64")
naive_data["earning_class"].value_counts()

binary_data = pd.get_dummies(naive_data)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns.difference(["earning_class"])], binary_data["earning_class"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("f64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("f64"))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb.classes_
gnb.class_prior_
gnb.class_count_
gnb.theta_
gnb.
y_pred = gnb.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))


'''
Decision Tree
'''
binary_data = pd.get_dummies(all_census_prep)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns.difference(["earning_class"])], binary_data["earning_class"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("f64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("f64"))


from sklearn.tree import DecisionTreeClassifier, export_graphviz
tree = DecisionTreeClassifier(criterion='entropy',max_depth=20)

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))

feature_names = list(X_train.columns)
export_graphviz(tree, out_file="tree.dot",feature_names=feature_names)

import pydotplus
import pyparsing
import StringIO
dotfile = StringIO.StringIO()
export_graphviz(tree, out_file=dotfile,feature_names=feature_names)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png("dtree2.png")

'''
Random Forests
'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
y_pred= rf.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))

'''
SVC
'''

from sklearn.svm import SVC
from sklearn import svm

clf4 = svm.SVC(kernel='rbf',C=6.0)
clf4.fit(X_train, y_train)
y_pred= clf4.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))

'''
KNearestNeighbors
'''
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)
y_pred= neigh.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))


'''
Corelación
'''
corr = binary_data.corr()
corr['age']
s = corr.abs().unstack()
s = s.order(ascending = False)



p = s.copy()

todel = []
for i in range(s.shape[0]):
    if s.axes[0][i][0] == s.axes[0][i][1]:
        todel.append(s.axes[0][i])
p = s.drop(todel,axis=0)    
p[p > 0.5]
binary_data = binary_data[binary_data.columns.difference(["sex_Female","workclass_missing","relationship_Spouse","race_White","marital_status_Never-married","native_country_United-States","relationship_Not-in-family","workclass_Private","race_Asian-Pac-Islander"])]



'''
PCA
'''

from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns.difference(["earning_class"])], binary_data["earning_class"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("f64")), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test.astype("f64")), columns=X_test.columns)

scores = []
f1 = []
components = []
var = []
for i in np.arange(0.05,1,0.05):
    pca = PCA(n_components = i)
    pca.fit(X_train)
    components.append(len(pca.explained_variance_ratio_))
    var.append(sum(pca.explained_variance_ratio_))
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    cls = linear_model.LogisticRegression()
    cls.fit(X_train_pca, y_train)
    y_pred = cls.predict(X_test_pca)
    cm = metrics.confusion_matrix(y_test, y_pred)
    scores.append((cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm)))
    f1.append(skl.metrics.f1_score(y_test, y_pred))

results = pd.DataFrame(zip(components,var,scores,f1))


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train_pca, y_train)
y_pred= neigh.predict(X_test_pca)

cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))
print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)

'''
Greedy Forward Search
'''

coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs = coefs.abs().order(ascending=False)
scores = []
f1 = []
for i in range(1,len(coefs)):
    
    cols = list(coefs[0:i].index)
    X_tra = X_train.copy()
    X_tes = X_test.copy()
    X_tra = X_tra[cols]
    X_tes = X_tes[cols]
    cls = linear_model.LogisticRegression()

    cls.fit(X_tra, y_train)
    y_pred = cls.predict(X_tes)
    cm = metrics.confusion_matrix(y_test, y_pred)
    scores.append((cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm)))
    f1.append(skl.metrics.f1_score(y_test, y_pred))

results = pd.DataFrame(zip(scores,f1))

'''
Selección de clasificador
'''

cols = list(coefs[0:30].index)
cols.append("earning_class")
binary_data = binary_data[cols]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns.difference(["earning_class"])], binary_data["earning_class"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("f64")), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test.astype("f64")), columns=X_test.columns)



from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

classifiers = []
classifiers.append(GaussianNB())
classifiers.append(linear_model.LogisticRegression())
for i in range(1,26,5):
    classifiers.append(KNeighborsClassifier(n_neighbors=i))
for i in range(10,100,10):
    classifiers.append(RandomForestClassifier(n_estimators=i))
for i in range(5,30,5):
    classifiers.append(DecisionTreeClassifier(criterion='entropy',max_depth=i))
for i in range(5,30,5):
    classifiers.append(DecisionTreeClassifier(criterion='gini',max_depth=i))
for i in range(1,21,5):
    classifiers.append(svm.SVC(kernel='rbf',gamma=i))

scores = []
f1 = []
for i in classifiers:
    print i
    i.fit(X_train, y_train)
    y_pred = i.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    scores.append((cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm)))
    f1.append(skl.metrics.f1_score(y_test, y_pred))

results = pd.DataFrame(zip(scores,f1))

tree = DecisionTreeClassifier(criterion='gini',max_depth=15)
tree.fit(X_train, y_train)
y_pred = tree.predict_proba(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))
skl.metrics.f1_score(y_test, y_pred)

y_pred[:,1]
from sklearn.metrics import roc_curve
len(tpr1)

fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred[:,1])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr1, tpr1, lw=1, label='Curva ROC')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(0,159,10):  
    la = round(thresholds1[xy],2)
    fi = (fpr1[xy],tpr1[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()

umbral1 = 0.33### Este umbral se selecciona de mis curvas ROC
Y_predict_coded1 = []
for i in y_pred[:,1]:
    if i > umbral1:
        Y_predict_coded1.append(1)
    else:
        Y_predict_coded1.append(0)
        
cm = metrics.confusion_matrix(y_test, Y_predict_coded1)
(cm[0][0]+cm[1][1]).astype('f64')/sum(sum(cm))
skl.metrics.f1_score(y_test, Y_predict_coded1)


print(skl.metrics.classification_report(y_test, Y_predict_coded1, target_names=["<50K",">50K"]))
