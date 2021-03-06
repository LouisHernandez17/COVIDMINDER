import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


Regions=pd.read_csv('data/csv/time_series/NY_counties_regions.csv',index_col='County')
Diabetes=pd.read_csv('data/csv/time_series/NY_counties_diabetes.csv',index_col='County')
Population=pd.read_csv('data/csv/time_series/NY_population.csv',index_col='County')
Test=pd.read_csv('data/csv/time_series/NY_county_tests.csv',index_col='County')
Covid=pd.read_csv('data/csv/time_series/covid_NY_counties.csv',index_col='county')
Palette=dict(Regions[['Region','Color']].to_dict('split')['data'])

Data=pd.DataFrame()
Data['Region']=Regions['Region']
Data[['Diabetes Rate','Food Insecure Rate']]=Diabetes[['pct_Adults_with_Diabetes','pct_Food_Insecure']]
Data['Fatality Rate']=Covid['deaths']/Covid['cases']
Data['Positive tests rate']=Test['Rate_positive']
Data['Population']=Population['Population']
Data['Cases per capita']=Test['Pct_positive']

Data=Data.dropna()
sns.set()
plt.figure()
g=sns.pairplot(Data,hue='Region',palette=Palette,vars=['Population','Diabetes Rate','Fatality Rate'])
g._legend.remove()
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",fontsize=8)
plt.show()
