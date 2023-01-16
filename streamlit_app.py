import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import streamlit as st

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

st.pyplot(fig)

x = df4['Cancer'][1:]
y = df4['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)



plt.title('Number of cancer for persons aged between 16 and 29 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

st.pyplot(fig)

x = df3['Cancer'][1:]
y = df3['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged between 30 and 44 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

st.pyplot(fig)

x = df2['Cancer'][1:]
y = df2['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged between 45 and 59 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

st.pyplot(fig)

x = df1['Cancer'][1:]
y = df1['Number'][1:]

fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

plt.title('Number of cancer for persons aged 60 years and more', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

st.pyplot(fig)

print("""The cancer that infected most :
<ul>
    <li>Lung</li>
    <li>Breast</li>
    <li>Leukaemia</li>
    <li>Thyroid</li>
</ul>

On this study we're going to focus on Lung and Breast (because Lung cancer ranked 1st two times and breast cancer because it targets the female gender specifically)""")

