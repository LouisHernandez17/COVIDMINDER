import os
import pandas as pd

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
        FullData_Cases['Confirmed ' + state] = data['Confirmed']
    if (data['Deaths'] > 100).any():
        FullData_Deaths['Deaths ' + state] = data['Deaths']

#%%
from Autoregressive import *

model=VAR(p=2,FullData=FullData_Cases,num_val=7)
model.fit()
model.predict(7,plot=True,savefig=True,path='CasesPredictionValidation')

#%%
DataArima=FullData_Deaths['Deaths Arizona'].values
dates=FullData_Deaths['Deaths Arizona'].index
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
model = ARIMA(DataArima, order=(p,d,q))
results=model.fit(disp=0)
print(results.summary())
fc, se, conf = results.forecast(10, alpha=0.05)
plt.figure()
plt.plot([dates[-1]+datetime.timedelta(days=i+1) for i in range(len(fc))],fc,'k--',label='forecast')
plt.plot([dates[-1]+datetime.timedelta(days=i+1) for i in range(len(fc))],conf[:,0],'k-.',label='Confidence interval')
plt.plot([dates[-1]+datetime.timedelta(days=i+1) for i in range(len(fc))],conf[:,1],'k-.')
plt.xticks(rotation=20)
plt.plot(dates,DataArima,'k-',label='actual')
plt.legend()
plt.show()

