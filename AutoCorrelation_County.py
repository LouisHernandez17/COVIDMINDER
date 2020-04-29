import datetime
import pandas as pd
import matplotlib.pyplot as plt

def Build_Data()
    Data=pd.read_csv('data/csv/time_series/covid_NY_TS_counties_long.csv',index_col=None)
    Data['date']=pd.to_datetime(Data['date'],yearfirst=True)
    Counties=pd.unique(Data['County'])
    Dates=pd.unique(Data['date'])
    Counties_Cases=pd.DataFrame()
    Counties_Deaths=pd.DataFrame()
    plt.figure()
    for county in Counties :
        CountyData=Data[Data['County']==county]
        CountyData=CountyData.set_index('date')
        CountyData=CountyData.asfreq('d')
        Counties_Cases[county]=CountyData['cases']
        Counties_Deaths[county]=CountyData['deaths']
    Counties_Deaths=Counties_Deaths.astype('float64')
    Counties_Cases=Counties_Cases.astype('float64'
    return (Counties_Deaths,Counties_Cases,Dates)
Counties_Deaths,Counties_Cases,Dates=Build_Data()
#plt.legend()
#%%
import numpy as np
plt.figure()
d=2
Data_Cases=Counties_Cases
Data_Deaths=Counties_Deaths
for i in range(d):
    Data_Cases=Data_Cases.diff()
    Data_Deaths=Data_Deaths.diff()

for county in Counties:
    if len(Data_Deaths[county].dropna())>30:
        index=np.linspace(-len(Data_Deaths[county].dropna())+1,len(Data_Deaths[county].dropna())-1,2*len(Data_Deaths[county].dropna())-1)
        corr=np.correlate(Data_Deaths[county].dropna(),Data_Cases[county].dropna(),mode='full')
        plt.plot(index,corr/np.max(np.abs(corr)),label=county)
        print(county,index[np.argmax(corr)])
plt.show()


#%%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
data=Counties_Cases
plt.figure()
k=4
for d in range(k):
    plt.subplot(k,1,d+1)
    plt.plot(data)
    plt.title(str(d)+' differentiations')
    data=data.diff().fillna(0)
d=2
data=Counties_Cases
for i in range(d):
    data=data.diff().fillna(0)
#plot_pacf(data['New York'])
plt.show()
p=3
#plot_acf(data['New York'])
q=1
plt.show()
plt.figure()
for county in Counties :
    try :
        if len(Counties_Cases[county].dropna())>10 and county!='Madison':
            print(county)
            model=SARIMAX(Counties_Cases[county].dropna(),order=(3,2,1))
            results=model.fit(disp=0)
            fc=results.forecast(10)
            pl=plt.plot(Counties_Cases[county].dropna().index,Counties_Cases[county].dropna())
            plt.plot([Counties_Cases[county].dropna().index[-1] + datetime.timedelta(days=i+1) for i in range(len(fc))],fc,'--',color=pl[0].get_color())
            #plt.plot([Counties_Cases[county].dropna().index[-1] + datetime.timedelta(days=i+1)for i in range(len(fc))],conf[:,1],'-.',color=pl[0].get_color())
            #plt.plot([Counties_Cases[county].dropna().index[-1] + datetime.timedelta(days=i+1)for i in range(len(fc))],conf[:,0],'-.',color=pl[0].get_color())
    except :
        print('Impossible')
plt.yscale('log')
plt.xticks(rotation=20)
plt.show()
#%%
model=SARIMAX(Counties_Cases['New York'].loc[Counties_Cases[['Westchester','Nassau']].dropna().index].fillna(method='pad'),exog=Counties_Cases[['Westchester','Nassau']].dropna(),order=(3,2,1))
results=model.fit(disp=0)
fc=results.forecast(1)
pl=plt.plot(Counties_Cases['New York'].dropna().index,Counties_Cases[county].dropna())
plt.plot([Counties_Cases['New York'].dropna().index[-1] + datetime.timedelta(days=i+1) for i in range(len(fc))],fc,'--',color=pl[0].get_color())
#plt.plot([Counties_Cases['New York'].dropna().index[-1] + datetime.timedelta(days=i+1)for i in range(len(fc))],conf[:,1],'-.',color=pl[0].get_color())
#plt.plot([Counties_Cases['New York'].dropna().index[-1] + datetime.timedelta(days=i+1)for i in range(len(fc))],conf[:,0],'-.',color=pl[0].get_color())
plt.yscale('log')
plt.xticks(rotation=20)
plt.show()
