# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:07:26 2020

@author: shara
"""

import pandas as pd
import numpy as np
import xlrd
import datetime
airlines = pd.read_excel("F:\\DS Assignments\\Forecasting\\Airlines_Data.xlsx")
airlines.head()
import datetime
month = datetime.datetime(airlines["Month"]).strftime('%B')
import calendar
calendar.month_name[3]

months = []
for i in range(0,96,1):
    months.append(airlines["Month"][i].month)    

months = calendar.month_name(months)

print(airlines["Month"][0].month)
len(airlines["Month"])

Months = []
for i in range(0,96,1):
    Months.append(calendar.month_name[months[i]])
    
airlines["Month_name"] = Months
# airlines = airlines.drop(columns=["Month"])
month_dummies = pd.DataFrame(pd.get_dummies(airlines["Month_name"]))
airlines1 = pd.concat([airlines,month_dummies], axis = 1)
airlines1["t"] = np.arange(1,97)
airlines1["t_squarred"] = airlines1['t'] * airlines1['t']
airlines1["log_pass"] = np.log(airlines1["Passengers"])
airlines1.Passengers.plot()
train = airlines1.head(85)
test = airlines1.tail(12)

import statsmodels.formula.api as smf
linear_model = smf.ols("Passengers ~ t", data = train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_linear))**2))
# RMSE_Linear = 53.31

Exp = smf.ols('log_pass ~ t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Passengers'])- np.array(np.exp(pred_Exp)))**2))
rmse_Exp

# RMSE_exp = 46.1

Quad = smf.ols('Passengers ~ t+t_squarred',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_squarred"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

# RMSE_QUAD = 48.61

add_sea = smf.ols('Passengers ~ January+February+March+April+May+June+July+August+September+October+November',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['January','February','March','April','May','June','July','August','September','October','November']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
# rmse_add_sea = 131

add_sea_Quad = smf.ols('Passengers ~ t+t_squarred+January+February+March+April+May+June+July+August+September+October+November',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['January','February','March','April','May','June','July','August','September','October','November','t','t_squarred']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
# rmse_add_sea_quad = 26.249

Mul_sea = smf.ols('log_pass~January+February+March+April+May+June+July+August+September+October+November',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# rmse_Mult_sea = 139

Mul_Add_sea = smf.ols('log_pass ~ t+January+February+March+April+May+June+July+August+September+October+November',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

# rmse_Mult_add_sea = 10

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
