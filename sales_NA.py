# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 21:51:09 2020

@author: shara
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import itertools
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
cola = pd.read_excel("F:\\DS Assignments\\Forecasting\\CocaCola_Sales_Rawdata.xlsx")
cola.head()
cola.Sales.plot()
cola["Quarter"][0][0:2]
len(cola)
cola["Quarter"][41]
a = "Q1"
b = "Q2"
c = "Q3"
d = "Q4"

quarters = []
for  i in range(42):
    if cola["Quarter"][i][0:2] == "Q1":
        quarters.append(a)
    elif cola["Quarter"][i][0:2] == "Q2":
        quarters.append(b)
    elif cola["Quarter"][i][0:2] == "Q3":
        quarters.append(c)
    else:
        quarters.append(d)
        
qtr_dummies = pd.DataFrame(pd.get_dummies(quarters))
cola1 = pd.concat([cola, qtr_dummies], axis = 1)
cola1["t"] = np.arange(1,43)
cola1["t_sq"] = cola1["t"] * cola1["t"]
cola1["log"] =np.log(cola1["Sales"])
train = cola1.head(36)
test = cola1.tail(6)
# add_sea_Quad = smf.ols('Sales ~ t+t_sq + Q1 + Q2 + Q3', data = train).fit()
# pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Q1',"Q2", "Q3",'t','t_sq']]))
# rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
# rmse_add_sea_quad 

# Mul_Add_sea = smf.ols('log ~ t+ Q1 + Q2 + Q3',data = train).fit()
# pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
# rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
# rmse_Mult_add_sea 

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing  
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
import seaborn as sns

cola1["Quarter"][0][3:5]
Year = []
for  i in range(42):
    Year.append(cola1["Quarter"][i][3:5])

cola1 = pd.concat([cola1, pd.DataFrame(Year)], axis = 1)    
cola1.rename(columns ={ cola1.columns[9]: "Year" }, inplace = True)
cola1 = pd.concat([cola1, pd.DataFrame(quarters)], axis = 1) 
cola1.rename(columns ={ cola1.columns[10]: "quarters" }, inplace = True)
from pylab import rcParams

heatmap_y_qtr = pd.pivot_table(data=cola1,values="Sales",index="Year",columns="quarters",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_qtr,annot=True,fmt="g")
sns.boxplot(x="quarters",y="Sales",data=cola1)
sns.boxplot(x="Year",y="Sales",data=cola1)

sns.lineplot(x="Year",y="Sales",hue="quarters",data=cola1)
cola1.Sales.plot(label="org")
for i in range(2,42,6):
    cola1["Sales"].rolling(i).mean().plot(label=str(i))

date = cola1['Quarter'].str.replace(r'(Q\d)_(\d+)', r'\2-\1')
date =  pd.PeriodIndex(date, freq='Q').to_timestamp()
date =  date - pd.DateOffset(years=100)
cola1 = pd.concat([cola1, pd.DataFrame(date)], axis = 1)
cola1 = cola1.rename(columns ={ cola1.columns[11]: "Time" })

cola1 = cola1.drop(columns ={ cola1.columns[0]})
cola1["Time"] = date
cola1 = cola1.drop(columns = ["Q1", 'Q2','Q3', 'Q4', 't', 't_sq',"log", 'Year', 'quarters'])
cola1 = cola1.set_index("Time")


decompose_ts_add = seasonal_decompose(cola1, model = "additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cola1,model="multiplicative")
decompose_ts_mul.plot()
model = sm.tsa.statespace.SARIMAX(cola1,order = [1,1,0], seasonal_order = (1,1,0,4), enforce_stationarity=False, enforce_invertibility=False)

results = model.fit()
print(results.summary().tables[1])

pred = results.get_prediction(start=pd.to_datetime("1895-01-01"), dynamic = False)
pred_ci = pred.conf_int()

forecast = pred.predicted_mean.to_frame('column_2').rename_axis('time').reset_index()
y_truth = cola1["1895-01-01":]
y_truth = pd.DataFrame(y_truth).reset_index()
mse = ((forecast["column_2"] - y_truth["Sales"]) * (forecast["column_2"] - y_truth["Sales"])).mean()
rmse = round((np.sqrt(mse)),2)

tsa_plots.plot_acf(cola.Sales,lags=10)
tsa_plots.plot_pacf(cola.Sales)

# train = cola1.head(36)
# test = cola1.tail(6)

# def MAPE(predicted,org):
#     temp = np.abs((pred-org))*100/org
#     return np.mean(temp)

# ses_model = SimpleExpSmoothing(train["Sales"]).fit()
# pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
# mean_squared_error(pred_ses,test.Sales)

# hw_model = Holt(train["Sales"]).fit()
# pred_hw = hw_model.predict(start = test.index[0],end = test.index[-1])
# mean_squared_error(pred_hw, test.Sales)


# hwe_model_add_add = ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
# pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])
# mean_squared_error(pred_hwe_add_add, test.Sales)

# hwe_model_mul_add = ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
# pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0],end = test.index[-1])
# mean_squared_error(pred_hwe_mul_add, test.Sales)
