import numpy as np
import pandas as pd
import plotly.figure_factory as ff
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

fig = ff.create_distplot(
        x, y)

st.plotly_chart(fig, use_container_width=True)

"""fig, ax = plt.subplots(figsize=(70,40))

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)


plt.title('Number of cancer for persons aged between 0 and 15 years', fontdict=font)
plt.xlabel('Cancer', fontdict=font)
plt.ylabel('Number', fontdict=font)

st.plotly_chart(fig)
"""
