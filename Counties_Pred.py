import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def Build_Data():
    Data = pd.read_csv('data/csv/time_series/covid_NY_TS_counties_long.csv', index_col=None)
    Data['date'] = pd.to_datetime(Data['date'], yearfirst=True)
    Counties = pd.unique(Data['County'])
    Dates = pd.unique(Data['date'])
    Counties_Cases = pd.DataFrame()
    Counties_Deaths = pd.DataFrame()
    plt.figure()
    for county in Counties:
        CountyData = Data[Data['County'] == county]
        CountyData = CountyData.set_index('date')
        CountyData = CountyData.asfreq('d')
        Counties_Cases[county] = CountyData['cases']
        Counties_Deaths[county] = CountyData['deaths']
    Counties_Deaths = Counties_Deaths.astype('float64')
    Counties_Cases = Counties_Cases.astype('float64')
    return (Counties_Deaths, Counties_Cases, Dates)


Counties_Deaths, Counties_Cases, Dates = Build_Data()
#%%
Regions = pd.read_csv('data/csv/time_series/NY_counties_regions.csv', index_col='County')
List_Regions=pd.unique(Regions['Region'])
Regions_Cases=pd.DataFrame()
for region in List_Regions:
    counties_reg=Regions[Regions['Region']==region].index
    counties=Counties_Cases.columns
    counties_region=[county for county in counties_reg if county in counties]
    Regions_Cases[region]=Counties_Cases[counties_region].sum(axis='columns')
Regions_Daily_Cases=Regions_Cases.diff()
Regions_Daily_Cases=Regions_Daily_Cases.rolling(7,min_periods=1).sum()
Values=[]
v=7
for region in List_Regions:
    Values=Values+list(Regions_Daily_Cases[region].iloc[:-v])+5*[np.nan]
plt.plot(Values)
plt.show()
#%%

from statsmodels.tsa.statespace.sarimax import SARIMAX



#%%
import warnings
import matplotlib.dates as mdates

Palette = dict(Regions[['Region', 'Color']].to_dict('split')['data'])
warnings.filterwarnings("ignore")
formatter = mdates.DateFormatter('%a %d/%m')
params=[]
scores=[]
plt.style.use('ggplot')
for p in range(1,5):
    for q in range(1,5):
        for d in range(3):
            try :
                model = SARIMAX(Values,order=(p,d,q), missing='drop', enforce_invertibility=False)
                results = model.fit(disp=0)
                scores_counties=[]
                plt.figure()
                ax = plt.gca()
                plt.xticks(rotation=20)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                ax.xaxis.set_major_formatter(formatter)
                for region in List_Regions :
                    DataCounty=Regions_Daily_Cases[region].dropna()
                    ModelCounty=SARIMAX(DataCounty[:-v],order=(p,d,q),missing='drop',enforce_invertibility=False)
                    res=ModelCounty.smooth(results.params)
                    fc=res.get_prediction(len(DataCounty)-v,len(DataCounty))
                    frame=fc.summary_frame(alpha=0.05)
                    fc=frame['mean']
                    Y=DataCounty.iloc[-v:].values
                    Yhat=fc[-v:].values
                    Ybar = np.mean(Y)
                    MAE = (sum(abs(Y - Yhat))/v)
                    scores_counties.append(MAE)
                    confInf=frame['mean_ci_lower']
                    confSup=frame['mean_ci_upper']
                    pl=plt.plot(DataCounty,label=region,color=Palette[region])
                    plt.fill_between(confInf.index, confSup, confInf, alpha=0.3,
                                     color=pl[0].get_color())
                    plt.title("Daily Cases Predicted with a single ARIMA({},{},{}) model".format(p,d,q))
                    plt.plot(fc,'--',color=pl[0].get_color())


                plt.text(1,0.9,'Mean Absolute Error : {:.0f}'.format(np.nanmean(scores_counties)),transform=ax.transAxes,horizontalalignment='left')
                plt.savefig('PredictionCountiesDaily/ARIMA{}{}{}_Pred.png'.format(p,d,q))
                #plt.xlim([DataCounty.iloc[-7:].index[0], DataCounty.iloc[-7:].index[-1]])
                plt.yscale('log')
                plt.legend(bbox_to_anchor=(1,0.5),loc='center left',fontsize=6)
                plt.show()
                scores.append(np.nanmean(scores_counties))
                params.append((p,d,q))
            except:
                print('Training Failed for parameters')
                print(p,d,q)
#%%
argbest=np.argmin(scores)
print('Best distance : ',scores[argbest])
print('Best params : ',params[argbest])
