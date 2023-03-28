import numpy as np
import pandas as pd
import matplotlib as mb
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA 
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from pylab import rcParams
plt.style.use('fivethirtyeight')

       
# Reading CSV file and indexing it based on time after parsing time as date
dateparse = lambda x: pd.datetime.strptime(x, '%YM%m')
data = pd.read_csv(r"C:\Users\baibcf\Desktop\sea_ice.csv",parse_dates=[0],date_parser=dateparse,index_col=0)

# creating two sperate dataframes for both region
data1= pd.DataFrame(data,columns= ['Arctic'])
data2= pd.DataFrame(data,columns= ['Antarctica'])

#function to decomepose time series to analyze trends, seasonality and residue
def trendAnalysis(data,number):
    x=number

    rcParams['figure.figsize'] = 11, 9
    decomposition = sm.tsa.seasonal_decompose(data, model='additive')
    
    fig = decomposition.plot()
    
    if x== 0:
         plt.title('trend and seasonality for Arctic' )
         
    else:
        plt.title('trend and seasonality for Antarctica' )
    plt.show()
    

#define KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
def kpssTest(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)

#define function for ADF (Augmented Dickey Fuller) test

def adfTest(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)


# function to visualize the observed time series
def plotFigure(dfindex,dfcolumn,number):
    plt.figure(figsize=(10,3), dpi=100)
    plt.plot(dfindex,dfcolumn, color='tab:red')
    plt.grid(True)
    rolmean = dfcolumn.rolling(12).mean()
    rolstd = dfcolumn.rolling(12).std()
    x=number
    
    if x== 0:
         plt.title('Ice melting range for Arctic' )
         
    else:
        plt.title('Ice melting range for Antarctica' )
    
    mean = plt.plot(rolmean, color='green', label='Rolling Mean')
    std = plt.plot(rolstd, color='orange', label = 'Rolling Std')
    plt.legend(loc='best')          
    plt.xlabel('Time')
    plt.ylabel('Range of variation')
    plt.show(block=False)


# SARIMA implementation and prediction
# SARIMA because data has seasonal component in it   
def predictMethod(dataframe):
    model =sm.tsa.statespace.SARIMAX(dataframe,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    
    #fitting the data
    model_fit = model.fit(disp=1)
    
    #plotting diagnostics
    model_fit.plot_diagnostics(figsize=(13, 12))
    plt.show()
    
    #forcasting 500 next predictions
    pred_uc = model_fit.get_forecast(steps=12)
    pred_ci = pred_uc.conf_int()
    
    #plotting observed forcast with actual values
    ax = dataframe.plot(label='observed', figsize=(13, 10))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rate of melting ice caps')
    
    plt.legend()
    plt.show()

    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())


def parameterEstimation(data):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for param in pdq:
        
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue


#Trend and seasonality check
print('#######Trend and seasonality check for arctic and anatractic##########')
trendAnalysis(data1,0)
trendAnalysis(data2,1)

#apply ADF (Augmented Dickey Fuller) Test on the series
adfTest(data1['Arctic'])

print('############################################################')
#apply  KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test test on the series
kpssTest(data1['Arctic'])

#optimal parameter estimation for SARIMAX using both dataframe 
print('############################################################')
parameterEstimation(data1)
print('############################################################')
parameterEstimation(data2)

print('############################################################')
# calling the plotting function               
plotFigure(data1.index,data1['Arctic'],0)
plotFigure(data2.index,data2['Antarctica'],1)

print('############################################################')
# calling the prediction function
predictMethod(data1)

print('############################################################')
predictMethod(data2)
