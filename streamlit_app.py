import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

df1 = pd.read_csv('cancer_60-and_more.csv')
df2 = pd.read_csv('cancer_45-59.csv')
df3 = pd.read_csv('cancer_30-44.csv')
df4 = pd.read_csv('cancer_15-29.csv')
df5 = pd.read_csv('cancer_0-15.csv')

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 40,
        }

x = df5['Cancer'][1:]
y = df5['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged between 0 and 15 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

plt.show()

x = df4['Cancer'][1:]
y = df4['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged between 16 and 29 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

plt.show()

x = df3['Cancer'][1:]
y = df3['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged between 30 and 44 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

plt.show()

x = df2['Cancer'][1:]
y = df2['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged between 45 and 59 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

plt.show()

x = df1['Cancer'][1:]
y = df1['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged 60 years and more', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

plt.show()

print("""The cancer that infected most :
<ul>
    <li>Lung</li>
    <li>Breast</li>
    <li>Leukaemia</li>
    <li>Thyroid</li>
</ul>

On this study we're going to focus on Lung and Breast (because Lung cancer ranked 1st two times and breast cancer because it targets the female gender specifically)""")

print("""<h2>Lung Cancer</h2>""")

indexLung_1 = (df1['Number'].where(df1['Cancer'] == 'Lung')).notna()
indexLung_2 = (df2['Number'].where(df2['Cancer'] == 'Lung')).notna()
indexLung_3 = (df3['Number'].where(df3['Cancer'] == 'Lung')).notna()
indexLung_4 = (df4['Number'].where(df4['Cancer'] == 'Lung')).notna()
indexLung_5 = (df5['Number'].where(df5['Cancer'] == 'Lung')).notna()

ratioLung_1 = df1[indexLung_1]['Number']*100/sum(df1['Number'][1:])
ratioLung_2 = df2[indexLung_2]['Number']*100/sum(df2['Number'][1:])
ratioLung_3 = df3[indexLung_3]['Number']*100/sum(df3['Number'][1:])
ratioLung_4 = df4[indexLung_4]['Number']*100/sum(df4['Number'][1:])
ratioLung_5 = df5[indexLung_5]['Number']*100/sum(df5['Number'][1:])

print("Ratio of cancer lungs from older range to younger range")
print([ratioLung_1,ratioLung_2,ratioLung_3,ratioLung_4,ratioLung_5])

print("""The younger we are the less likely we're to developp a Lung cancer and the reasons are simple, to quote https://www.cdc.gov/cancer/lung/basic_info/index.htm#:~:text=Cigarette%20smoking%20is%20the%20number,family%20history%20of%20lung%20cancer.: "Cigarette smoking is the number one cause of lung cancer. Lung cancer also can be caused by using other types of tobacco (such as pipes or cigars), breathing secondhand smoke, being exposed to substances such as asbestos or radon at home or work, and having a family history of lung cancer"
The elders are most likely to have spent more time smoking thus more chances to develop this type of cancer.
To prevent this cancer before its detection, we should just stop smoking""")

df_lung = pd.read_csv('lung_cancer.csv')
df_lung = df_lung.drop(columns = ['Unnamed: 26','Unnamed: 27'])
print(df_lung)

corr_matrix = df_lung.corr()
print(corr_matrix['Result'])

As we can see, Age is not sufficient and no other variable on its own is sufficient to explain the fact to have lung cancer.</br>

print("""Using "Multi-class classification" a machine learning we'll try to predict if a person has cancer or not </br>
for this I chose 3 variables to explain using 4 variables : 
<ul>
    <li>Age</li>
    <li>Smoking</li>
    <li>Obesity</li>
    <li>Genetic Risk</li>
</ul>""")

df_multiple_lung = df_lung[['Age','Smoking','Obesity','Genetic Risk','Result']]
print(df_multiple_lung)


X = df_multiple_lung[['Age','Smoking','Obesity','Genetic Risk']]
y = df_multiple_lung['Result']
  
print("dividing X, y into train and test data (keep 60% for the train size and 40 for the test siez)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state = 0)

print("Let's train our model")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)

print("Let's get our prediction")
knn_predictions = knn.predict(X_test)
knn_predictions

print(Now we see how accurate was our model)
accuracy = knn.score(X_test, y_test)
accuracy

print("""<h2>Breast Cancer</h2>""")

indexBreast_1 = (df1['Number'].where(df1['Cancer'] == 'Breast')).notna()
indexBreast_2 = (df2['Number'].where(df2['Cancer'] == 'Breast')).notna()
indexBreast_3 = (df3['Number'].where(df3['Cancer'] == 'Breast')).notna()
indexBreast_4 = (df4['Number'].where(df4['Cancer'] == 'Breast')).notna()
indexBreast_5 = (df5['Number'].where(df5['Cancer'] == 'Breast')).notna()

ratioBreast_1 = df1[indexBreast_1]['Number']*100/sum(df1['Number'][1:])
ratioBreast_2 = df2[indexBreast_2]['Number']*100/sum(df2['Number'][1:])
ratioBreast_3 = df3[indexBreast_3]['Number']*100/sum(df3['Number'][1:])
ratioBreast_4 = df4[indexBreast_4]['Number']*100/sum(df4['Number'][1:])
ratioBreast_5 = df5[indexBreast_5]['Number']*100/sum(df5['Number'][1:])

print("Ratio of breast cancer from older range to younger range")
print([ratioBreast_1,ratioBreast_2,ratioBreast_3,ratioBreast_4,ratioBreast_5])

df_breast = pd.read_csv('breast_cancer.csv')
df_breast = df_breast.drop(columns = ['id'])
print(df_breast)

print("""M = malignant (cancerous) </br>
B = benign(non cancerous) </br>
We're gonna transform these data into 0 for non cancerous and 1 for cancerous""")

l=LabelEncoder()
df_breast['diagnosis']=l.fit_transform(df_breast.diagnosis)
print(df_breast)

x = df_breast.drop('diagnosis',axis=1)
y = df_breast['diagnosis']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

svm_model_linear = SVC(kernel = 'linear').fit(xtrain, ytrain)
svm_predictions = svm_model_linear.predict(xtest)

print(svm_predictions)

print("Let's test the accuracy of our model")
accuracy = svm_model_linear.score(xtest, ytest)
print(accuracy)

print("Having an accuracy of 95%, we can conclude that breast cancer can be predicted using all the variables in our dataframe (without Age because it's not a good enough variable to predict breast cancer), but through a early diagnosis we can predict an find solutions to do so")


