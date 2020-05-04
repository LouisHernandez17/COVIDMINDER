import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_files = [i[0] + '/' + files for i in os.walk('data/csv', False) for files in i[2] if files[:2].isnumeric()]
d0 = pd.read_csv(all_files[0])
d0 = d0[d0['Country_Region'] == 'US']
States = list(pd.unique(d0['Province_State']))
States.remove('Recovered')
States_Deaths=[]
States_Cases=[]
FullData_Cases=pd.DataFrame()
FullData_Deaths=pd.DataFrame()
states_to_remove=[]
for state in States:
    data = pd.DataFrame(columns=['Confirmed', 'Deaths', 'Recovered', 'Active', 'Date'])
    for file in all_files:
        data_state = pd.read_csv(file)
        data_state = data_state[data_state['Province_State'] == state]
        data_state = data_state[['Confirmed', 'Deaths', 'Recovered', 'Active']]
        data_state = data_state.sum()
        date = file[9:-4]
        data_state['Date'] = date
        data = data.append(data_state, ignore_index=True)
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index, format="%m-%d-%Y")
    data = data.asfreq('d')
    data=data.apply(pd.to_numeric)
    data = data.interpolate(method='linear')
    if (data['Confirmed'] > 1000).all():
        FullData_Cases[state] = data['Confirmed']
    if (data['Deaths'] > 100).any():
        FullData_Deaths[state] = data['Deaths']

#%%
from Autoregressive import *

model=VAR(p=1,FullData=FullData_Cases,num_val=7)
model.fit()
score=model.predict(7,plot=True,score=True)
print("Mean score Cases : {:.3f}".format(np.mean(score)))

model=VAR(p=1,FullData=FullData_Deaths,num_val=7)
model.fit()
score=model.predict(7,plot=True,score=True)
print("Mean score Deaths : {:.3f}".format(np.mean(score)))
#%%
DataArima=FullData_Deaths['Arizona'].values
dates=FullData_Deaths['Arizona'].index
Diff=[DataArima]
plt.figure()
for i in range(4):
    plt.subplot(4,2,2*i+1)
    plt.plot(Diff[i])
    plt.xlim(0, len(Diff[i]))
    plt.title('Data after {} differentiations'.format(i))
    plt.subplot(4,2,2*i+2)
    plt.xlim(-0.5,len(Diff[i]))
    plt.xcorr(Diff[i],Diff[i])
    plt.title('Autocorrelation')
    Diff.append(Diff[i][1:]-Diff[i][:len(Diff[i])-1])
plt.show()
d=2
#%%
DataArimaDiff=Diff[d]
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
plt.figure()
plot_pacf(DataArimaDiff)
plt.title("Partial Autocorrelation")
plt.show()
p=1
plt.figure()
plot_acf(DataArimaDiff)
plt.title("Autocorrelation")
plt.show()
q=1
#%%
from statsmodels.tsa.arima_model import ARIMA
print('plop')
for p in range(4):
    for q in range(4):
        scores=[]
        print('-----------')
        print('p=',p)
        print('q=',q)
        for state in FullData_Deaths.columns:
            try :
                DataTrain=FullData_Deaths[state].iloc[:-7]
                dates=FullData_Deaths.index
                DataValidation=FullData_Deaths[state].iloc[-7:]
                model = ARIMA(DataTrain, order=(p,d,q))
                results=model.fit(disp=0)
                fc, se, conf = results.forecast(7, alpha=0.05)
                Y = DataValidation.values
                Yhat = fc
                Ybar= np.mean(Y)
                R2=1-sum((Y-Yhat)**2)/sum((Y-Ybar)**2)
                scores.append(R2)
            except :
                ()
                #print(state,'Impossible to estimate')
            #     plt.figure()
            #     plt.plot(dates[:len(DataTrain)], DataTrain, '-', label='actual')
            #     plt.plot(dates[len(DataTrain):],fc,'--',label='forecast')
            #     plt.plot(dates[len(DataTrain):],DataValidation,'-',label='Validation')
            # plt.xticks(rotation=20)
            # plt.legend()
            #plt.show()
        print('score', np.mean(scores))
        print(len(scores))
