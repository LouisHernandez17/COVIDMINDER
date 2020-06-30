import matplotlib.pyplot as plt
import pandas as pd

DataPlot = pd.read_csv('data/csv/time_series/covid_NY_TS_plot.cases.csv', parse_dates=True)
plt.style.use('ggplot')
DataPlot.set_index(['date', 'County'], inplace=True)
# %%
Regions = pd.read_csv('data/csv/time_series/NY_counties_regions.csv', index_col='County')
List_Regions = pd.unique(Regions['Region'])
Regions_Cases = pd.DataFrame()
for region in List_Regions:
    DataRegion = DataPlot[DataPlot['Region'] == region]
    DataRegion = DataRegion.sum(level=0)
    DataRegion['p_cases'] = DataRegion['cases'] / DataRegion['Population'] * 100000
    Regions_Cases[region] = DataRegion['p_cases']
Regions_Cases.index = pd.to_datetime(Regions_Cases.index)
Regions_Daily_Cases = Regions_Cases.diff()
Regions_Daily_Cases = Regions_Daily_Cases.fillna(0)
Regions_Daily_Cases = Regions_Daily_Cases.rolling(7, min_periods=1).sum()
#%%
from sklearn.cluster import  KMeans
import numpy as np

X=Regions_Daily_Cases.values.transpose()
kmeans=KMeans(n_clusters=3).fit(X)
colors=['red','blue','green']
plt.figure()
for i in range(len(List_Regions)) :
    region=List_Regions[i]
    ind=kmeans.labels_[i]
    plt.plot(Regions_Daily_Cases[region],color=colors[ind],label=region)
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=6)
#plt.yscale('log')
plt.title('Clustering with 3 clusters')
plt.show()
for i in range(len(X)):
    X[i,:]=X[i,:]/np.max(X[i,:])
kmeans=KMeans(n_clusters=2).fit(X)
colors=['red','blue','green']
#%%
plt.figure()
for i in range(len(List_Regions)) :
    region=List_Regions[i]
    ind=kmeans.labels_[i]
    color=[0.0,0.0,0.0]
    color[ind] =1-0.8*(i+1)/(len(List_Regions)+1)
    color=tuple(color)
    plt.plot(Regions_Daily_Cases[region],color=color,label=region)
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=6)
#plt.yscale('log')
plt.title('Clustering with 2 clusters')
plt.show()

%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1,2,3],color=(0.1,0.3,0.2))
plt.show()

