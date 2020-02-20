#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importieren von Bibliotheken
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from pyomo.opt import SolverFactory
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Ablageort von Dateien
#HIER BITTE ÄNDERN!!!
directory = '...'


# In[3]:


data = pd.DataFrame()
file = {}
price_data = {}
file_start = "intraday_transactions_germany_austria_"
file_end = "_export.csv"
for i in range(3):
    year = 2015 +i
    file[i] =directory+ file_start + str(year) + file_end
    price_data[i] = pd.read_csv(file[i], sep = ",", decimal =".")
    price_data[i].index = price_data[i]["Date"]
    price_data[i] = price_data[i].drop(price_data[i].columns[0:3], axis = 1)
    data = data.append(price_data[i])


# In[4]:


tage ={}
for j in range(1,2):#len(data.index.unique())):
    date = data.index.unique()[j]
    tage[j] = pd.DataFrame(columns = data.columns)
    tage[j] = data[data.index == date]


# In[5]:


tage


# In[6]:


data_auctions = pd.DataFrame()
file = {}
auction_price_data = {}
file_start = "intraday_auction_spot_prices_15-call-DE_"
file_end = ".csv"
for i in range(3):
    year = 2015 +i
    file[i] =directory+ file_start + str(year) + file_end
    auction_price_data[i] = pd.read_csv(file[i], sep = ",", decimal =".", parse_dates = True)
    auction_price_data[i]["Delivery day"] = pd.to_datetime(auction_price_data[i]["Delivery day"])
    auction_price_data[i].index = auction_price_data[i]["Delivery day"]
    auction_price_data[i] = auction_price_data[i].drop(auction_price_data[i].columns[0:1], axis = 1)
    data_auctions = data_auctions.append(auction_price_data[i])
    data_auctions = data_auctions.sort_index()


# In[7]:


auction_tage ={}
for j in range(1,2):#len(data_auctions.index.unique())):
    date = data_auctions.index.unique()[j]
    auction_tage[j] = pd.DataFrame(columns = data_auctions.columns)
    auction_tage[j] = data_auctions[data_auctions.index == date]
    auction_tage[j] = auction_tage[j].dropna(axis=1)


# In[8]:


##Base data
#Capacity of battery in kWh
kap_speicher = 100
#Maximum power charge
max_power = 50
#Minimum power charge
min_power = 0
#Maximum power discharge
neg_max_power = 0
#Minimum power discharge
neg_min_power = -50
#Initial soc
init_speicher = 50
#Final soc
fSOC = 50
# #Lead time [15-min.-intervalls]
# leadtime=10*4
# #Horizon of optimization [15-min.-intervalls]
horizon = 96
#charging efficiency
ladewirkungsgrad = 0.9
#discharging efficiency
entladewirkungsgrad = 0.9
# #start time of charge
# start_time = EV_data['Start_15'].loc[0:100]
# #Calculate time of forecast = time of initial optimization 
# prognosezeitpunkt = start_time - datetime.timedelta(hours=leadtime/4)
# #End Time of charge
# end_time = EV_data['Ende_15'].loc[0:100]
#trading fee per MWh
trading_fee = 0.09
grid_fee = 7.21


# In[9]:


j = 1
columns = auction_tage[j].columns[0:96]
for j in range(1,2):#len(data.index.unique())):
    prices = auction_tage[j].iloc[:,0:96] 
    prices = prices.reset_index(drop = True)
    tage[j] = tage[j][tage[j]['product']=='VSTD']
    tage[j] = tage[j].reset_index(drop = True)
    for i in range(len(tage[j].index)):
        new_data = pd.DataFrame(prices[-1:].values, index= [i+1], columns=prices.columns)
        prices = prices.append(new_data)
        prices.loc[i+1][(tage[j]["VSTD.from"][i]-1):tage[j]["VSTD.to"][i]]=tage[j]["Price..EUR."][i]


# In[10]:


revenues ={}


# In[ ]:


import matplotlib.patches as mpatches
import matplotlib.lines as mlines
directory = ''
import pyomo.environ as pyo
##Initial optimization
#Data for missing prices
i = 0
#Data Frame for price data for Optimization
preise_buysell = prices.iloc[0,:]
min_price = min(preise_buysell)
max_price = max(preise_buysell)
#Prices for sell/buy
preise_buysell = preise_buysell.reset_index(drop = True)
preise_buysell = pd.DataFrame({'preise': preise_buysell.values})
preise_buysell.to_csv(directory + 'buysell_preise.csv',index_label = "t")
#Energy values that restrict charge (here only used final and initial SOC)
energy_need = preise_buysell.copy()
energy_need[:] = 0
energy_need= energy_need.rename(columns ={'preise': 'energy'})
energy_need['energy'][horizon-1] =fSOC
energy_need['energy'][0]=init_speicher
energy_need.to_csv(directory + 'energy_need.csv',index_label = "t")
#Power values charge = existing schedule for charging
p_supply = preise_buysell.copy()
p_supply[:] = 0
p_supply= p_supply.rename(columns ={'preise': 'Power'})
p_supply['Power'][:] =0
p_supply.to_csv(directory + 'p_supply.csv',index_label = "t")
#Power values discharging = existing schedule for discharging
p_withdraw = preise_buysell.copy()
p_withdraw[:] = 0
p_withdraw= p_withdraw.rename(columns ={'preise': 'Power'})
p_withdraw.to_csv(directory + 'p_withdraw.csv',index_label = "t")


#Definition of optimization model
model = pyo.AbstractModel()

#Sets and parameters of the abstract model
#Sets = Indices
model.t = pyo.Set(dimen = 1) #time periods for trade
#Parameter = exogenous variables (input variables)
model.preis_buysell = pyo.Param(model.t)
model.energy_need = pyo.Param(model.t)
model.p_supply = pyo.Param(model.t)
model.p_withdraw = pyo.Param(model.t)
# model.preis_exis = pyo.Param(model.t)

#Variables of the abstract model (= decision variables)
model.p_buy = pyo.Var(model.t, domain = pyo.Reals, bounds = (0, max_power-neg_min_power), initialize=0)
model.p_sell = pyo.Var(model.t, domain = pyo.Reals, bounds = (neg_min_power-max_power, 0), initialize=0)
model.soc = pyo.Var(model.t, domain= pyo.NonNegativeReals, bounds = (0, kap_speicher), initialize = 0)
model.p_result_pos = pyo.Var(model.t, domain = pyo.NonNegativeReals, initialize=0)
model.p_result_neg = pyo.Var(model.t, domain = pyo.NonPositiveReals, initialize=0)
model.buy = pyo.Var(model.t, domain = pyo.Binary,initialize = 0)
model.sell = pyo.Var(model.t, domain = pyo.Binary, initialize = 0)
model.charge = pyo.Var(model.t, domain = pyo.Binary, initialize = 0)
model.discharge = pyo.Var(model.t, domain = pyo.Binary, initialize = 0)

#Objective function of the abstract model
def obj_expression(model):
    return 1/4*1/1000*sum(model.p_buy[t]*model.preis_buysell[t] + 
                          model.p_sell[t]*model.preis_buysell[t] + 
                          model.p_buy[t]*trading_fee - 
                          model.p_sell[t]*trading_fee  
                          for t in model.t)
model.OBJ = pyo.Objective(rule=obj_expression)

#Schedule of EV 
def resulting_power_rule(model,t):
    if t == horizon-1:
        return model.p_result_pos[t] + model.p_result_neg[t] == 0
    else:
        return model.p_result_pos[t] + model.p_result_neg[t]  == model.p_supply[t]+model.p_buy[t]+        model.p_withdraw[t]+model.p_sell[t]
model.resulting_power_rule = pyo.Constraint(model.t, rule=resulting_power_rule)

#Binary variable buy to avoid buying and selling in same time step
def buy_rule(model,t):
    if t == horizon-1:
        return model.p_buy[t] == 0
    else:
        return model.p_buy[t] <= model.buy[t]*(max_power-neg_min_power)/ladewirkungsgrad
model.buy_rule = pyo.Constraint(model.t, rule=buy_rule)

#Binary variable sell to avoid buying and selling in same time step
def sell_rule(model,t):
    if t == horizon-1:
        return model.p_sell[t] == 0
    else:
        return model.p_sell[t] >= model.sell[t]*(neg_min_power-max_power)/entladewirkungsgrad
model.sell_rule = pyo.Constraint(model.t, rule=sell_rule)

# #binary variable buy II to avoid trading if no trade existed
# def buy_rule2(model,t):
#     return model.p_buy[t] <= model.preis_exis[t]*(max_power-neg_min_power)
# model.buy_rule2 = pyo.Constraint(model.t, rule=buy_rule2)

# #binary variable sell II to avoid trading if no trade existed
# def sell_rule2(model,t):
#     return model.p_sell[t] >= model.preis_exis[t]*(neg_min_power-max_power)
# model.sell_rule2 = pyo.Constraint(model.t, rule=sell_rule2)

#constraint to avoid buying and selling in same time step
def buysell_rule(model,t):
    return model.buy[t]+model.sell[t] <= 1
model.buysell_rule = pyo.Constraint(model.t, rule=buysell_rule)

# #binary variable to limit minimum charging
# def charge_rule(model,t):
#     return model.p_result_pos[t] >= model.charge[t]*min_power
# model.charge_rule = pyo.Constraint(model.t, rule=charge_rule)

#binary variable to limit maximum charging
def charge_rule2(model,t):
    return model.p_result_pos[t] <= model.charge[t]*max_power
model.charge_rule2 = pyo.Constraint(model.t, rule=charge_rule2)

#binary variable to limit minimum discharging 
def discharge_rule(model,t):
    return model.p_result_neg[t] <= model.discharge[t]*neg_max_power
model.discharge_rule = pyo.Constraint(model.t, rule=discharge_rule)

#binary variable to limit maximum discharging
def discharge_rule2(model,t):
    return model.p_result_neg[t] >= model.discharge[t]*neg_min_power
model.discharge_rule2 = pyo.Constraint(model.t, rule=discharge_rule2)

#EV SOC
def soc_rule(model,t):
    if t == 0:
        return model.soc[t] == model.energy_need[t]
    if t >= 1 and t <= horizon:
        return model.soc[t] == model.soc[t-1]+     1/4*model.p_result_pos[t-1]*ladewirkungsgrad+     1/4*model.p_result_neg[t-1]/entladewirkungsgrad
    return pyo.Constraint.Skip
model.soc_rule = pyo.Constraint(model.t, rule=soc_rule)

#MIN SOC
def min_soc_rule(model,t):
    return model.soc[t] >= model.energy_need[t]
model.min_soc_rule = pyo.Constraint(model.t, rule=min_soc_rule)

#Open DataPortal 
data = pyo.DataPortal() 

#Read all the data from different files
data.load(filename='buysell_preise.csv',format='set', set='t')
data.load(filename='buysell_preise.csv',index='t', param='preis_buysell')
data.load(filename='energy_need.csv',index='t', param='energy_need')
data.load(filename='p_supply.csv',index='t', param='p_supply')
data.load(filename='p_withdraw.csv',index='t', param='p_withdraw')
# data.load(filename='preis_exis.csv', index = 't', param = 'preis_exis')
instance = model.create_instance(data)
#Use solver gurobi
opt = SolverFactory('gurobi')
#Change mipgab to 5%
opt.options['mipgap'] = 0.05
model.pprint()
#Solve
results = opt.solve(instance) 

# Storing of results in CSV
name = "results_ev_v0.csv"
f = open(name, 'w')
f.write("t" + ", ")
for t in instance.t.value:
    f.write(str(t)+", ")
f.write("\n")
f.write("Power Charge [kW]"+", ")
for t in instance.t.value:
    f.write(str(instance.p_supply[t]) + ", ")
f.write("\n")
f.write("Power Buy [kW]"+", ")
for t in instance.t.value:
    f.write(str(instance.p_buy[t].value) + ", ")
f.write("\n")
f.write("Power Discharge [kW]"+", ")
for t in instance.t.value:
    f.write(str(instance.p_withdraw[t]) + ", ")
f.write("\n")
f.write("Power Sell [kW]"+", ")
for t in instance.t.value:
    f.write(str(instance.p_sell[t].value) + ", ")
f.write("\n")
f.write("Resulting Power [kW]"+", ")
for t in instance.t.value:
    f.write(str(instance.p_result_pos[t].value+ instance.p_result_neg[t].value) + ", ")
f.write("\n")
f.write("Energy [kWh]"+", ")
for t in instance.t.value:
    f.write(str(instance.soc[t].value) + ", ")
f.write("\n")
f.write("Current Price [EUR/MWh]"+", ")
for t in instance.t.value:
    f.write(str(instance.preis_buysell[t]) + ", ")
f.write("\n")
# f.write("Existierender Preis [1/0]"+", ")
# for t in instance.t.value:
#     f.write(str(instance.preis_exis[t]) + ", ")
# f.write("\n")
f.close()

#Dataframe with results
results_power = pd.read_csv(name, index_col = 't',  sep = ",", error_bad_lines=False)
results_power = results_power.T
results_power = results_power[:-1]
results_power = results_power.astype('float')
date_list = [auction_tage[j].index[0] + datetime.timedelta(minutes=15*x) for x in range(0, 96)]
results_power['Date']= date_list
results_power['Date']=results_power['Date'].apply(lambda x: x.strftime('%H:%M'))
results_power['Resulting Power [kW] (Old)']=results_power['Power Charge [kW]']+results_power['Power Discharge [kW]']
#Dataframe with power and energy values
primary = results_power[['Date','Resulting Power [kW]','Energy [kWh]']].copy()
primary = primary.rename(columns={"Resulting Power [kW]": "Current schedule [kW]"})
primary = primary.set_index("Date")
#Dataframe with prices
secondary = results_power[['Date','Current Price [EUR/MWh]']].copy()
secondary = secondary.rename(columns={"Current Price [EUR/MWh]": "Price of current optimization [EUR/MWh]"})
#Dataframe with buy- and sell power
tertiary = results_power[['Date','Power Buy [kW]', 'Power Sell [kW]']].copy()
tertiary= tertiary.rename(columns={"Power Buy [kW]": "Power purchased [kW]", 
                                   "Power Sell [kW]": "Power sold [kW]"})
#Plot
fig, ax1 = plt.subplots()
ax1 = primary['Current schedule [kW]'].plot(x = 'Date', color="black", lineWidth = 2)
ax2 = secondary.plot(x = 'Date',color = 'red', secondary_y=True, ylim =(neg_min_power,max(kap_speicher, max_power)+1), ax=ax1, linestyle = "dashed")
tertiary.plot(x = 'Date', figsize=(13,6.5),kind='bar',ax=ax1, color =("darkviolet", "pink"),width = 1)
primary['Energy [kWh]'].plot(x = 'Date', ax = ax1)
ax2.set_ylim(min_price,max_price)
title = "Initial Optimization based on Intraday Auction"
plt.title(title)
ax1.set_xlabel('Time [hh:mm]', fontsize = 16)
ax1.set_ylabel('Power [kW]/ Energy [kWh]', fontsize = 16)
ax2.set_ylabel('Energy price [€/MWh]', fontsize = 16,color = 'red')
#Plot von horizontalen Achsen
ax1.axhline(y = 0, color = "k")
# ax2.axhline(y = 0, color = "b")
#Legendenplot
ax1.legend(bbox_to_anchor=(0,1.05,1,0.15), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
ax2.legend(bbox_to_anchor=(0,1.10,1,0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
ax2.set_ylim(-50,80)
ax1.set_ylim(-20,100)

ax2.tick_params(axis='y', colors='red')

n = 12
for index, label in enumerate(ax1.get_xticklabels()):
    label.set_rotation(90)
    if index % n != 0:
        label.set_visible(False)

#Speicherung von Graphen
fname = 'figure' + str("%04d" % i)
plt.savefig(fname)
plt.close()
plt.close()

#Creation of cost vector
profits = pd.Series(len(tage[j].index)+1)
#Calculation of costs
profits[0] = 0
#Creation of vector for number of trades
trades = pd.Series(len(tage[j].index)+1)
#Calculation of number of trades
trades[0] = sum(results_power['Power Buy [kW]'] != 0) + sum(results_power['Power Sell [kW]'] != 0)
#Series for schedule
fahrplan = pd.Series(range(96))
#Series for SOC development
soc = pd.Series(range(96))
last_minute = 0
print_counter = 0
minute = 0

##Optimization before charge  
for i in range(1,len(tage[j].index)):
    #Change prices for next optimization
    
    if tage[j]["Time.Stamp"][i] >= (auction_tage[j].index - datetime.timedelta(minutes = 5)):
        #Substract 5 minutes due to gate closure time
        minute = int((pd.to_datetime(tage[j]["Time.Stamp"][i]) + datetime.timedelta(minutes = 5)).hour*4
        +(pd.to_datetime(tage[j]["Time.Stamp"][i])+ datetime.timedelta(minutes = 5)).minute/15)
        
        if minute >= last_minute:
            energy_need.loc[minute+1] = results_power["Energy [kWh]"][minute+1]
            energy_need[minute+1:].to_csv(directory + 'energy_need.csv',index_label = "t")

            #EV SOC
            def soc_rule(model,t):
                if t == minute+1:
                    return model.soc[t] == model.energy_need[t]
                if t > minute+1 and t <= horizon:
                    return model.soc[t] == model.soc[t-1] + 1/4*model.p_result_pos[t-1]*ladewirkungsgrad + 1/4*model.p_result_neg[t-1]/entladewirkungsgrad
                return pyo.Constraint.Skip
            model.soc_rule = pyo.Constraint(model.t, rule=soc_rule)
            
            last_minute = last_minute +1

#         fahrplan = results_power['Resulting Power [kW]'][:minute+1]
#         soc = results_power['Energy [kWh]'][:minute+1]
            
        #Update supply power and discharge power
        power = results_power["Resulting Power [kW]"]
        mask1 = power >= 0
        mask2 = power <= 0
        p_supply['Power'][:] = power*mask1
        p_supply[minute+1:].to_csv(directory + 'p_supply.csv',index_label = "t")
        p_withdraw['Power'][:] = power*mask2
        p_withdraw[minute+1:].to_csv(directory + 'p_withdraw.csv',index_label = "t")
        preise_buysell = pd.DataFrame({'preise': prices.loc[i][0:].values})
        preise_buysell[minute+1:].to_csv(directory + 'buysell_preise.csv',index_label = "t")
        
        
#         last_minute = minute
    
    
    else:
        energy_need['energy'][minute]= init_speicher
        #Update supply power and discharge power
        power = results_power["Resulting Power [kW]"]
        mask1 = power >= 0
        mask2 = power <= 0
        p_supply['Power'][:] = power*mask1
        p_supply.to_csv(directory + 'p_supply.csv',index_label = "t")
        p_withdraw['Power'][:] = power*mask2
        p_withdraw.to_csv(directory + 'p_withdraw.csv',index_label = "t")
        #Update prices
        preise_buysell = pd.DataFrame({'preise': prices.loc[i][0:].values})
        preise_buysell.to_csv(directory + 'buysell_preise.csv',index_label = "t")
        energy_need['energy'][horizon]=fSOC
        energy_need.to_csv(directory + 'energy_need.csv',index_label = "t")


    #Open data portal
    data = pyo.DataPortal() 

    #Read all the data from different files
    data.load(filename='buysell_preise.csv',format='set', set='t')
    data.load(filename='energy_need.csv',index='t', param='energy_need')
    data.load(filename='buysell_preise.csv',index='t', param='preis_buysell')
    data.load(filename='p_supply.csv',index='t', param='p_supply')
    data.load(filename='p_withdraw.csv',index='t', param='p_withdraw')
    # data.load(filename='preis_exis.csv', index = 't', param = 'preis_exis')
    instance = model.create_instance(data)
    #Use solver gurobi
    opt = SolverFactory('gurobi')
    #Change mipgab to 5%
    opt.options['mipgap'] = 0.05
#     model.pprint()
    #Solve
    results = opt.solve(instance) 
    print(instance.OBJ())
    profits[i] = profits[i-1] + min(instance.OBJ(),0)
    
    if (instance.OBJ()<-0.000001):
        print_counter = print_counter +1
        
        trades[i] = sum(results_power['Power Buy [kW]'] != 0) + sum(results_power['Power Sell [kW]'] != 0)


        # Storing of results in CSV
        name = "results_ev.csv"
        f = open(name, 'w')
        f.write("t" + ", ")
        for t in instance.t.value:
            f.write(str(t)+", ")
        f.write("\n")
        f.write("Power Charge [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_supply[t]) + ", ")
        f.write("\n")
        f.write("Power Buy [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_buy[t].value) + ", ")
        f.write("\n")
        f.write("Power Discharge [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_withdraw[t]) + ", ")
        f.write("\n")
        f.write("Power Sell [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_sell[t].value) + ", ")
        f.write("\n")
        f.write("Resulting Power [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_result_pos[t].value+ instance.p_result_neg[t].value) + ", ")
        f.write("\n")
        f.write("Energy [kWh]"+", ")
        for t in instance.t.value:
            f.write(str(instance.soc[t].value) + ", ")
        f.write("\n")
        f.write("Current Price [EUR/MWh]"+", ")
        for t in instance.t.value:
            f.write(str(instance.preis_buysell[t]) + ", ")
        f.write("\n")
        # f.write("Existierender Preis [1/0]"+", ")
        # for t in instance.t.value:
        #     f.write(str(instance.preis_exis[t]) + ", ")
        # f.write("\n")
        f.close()
        
        fahrplan = results_power['Resulting Power [kW]'][:minute+1]
        soc = results_power['Energy [kWh]'][:minute+1]
        
        
        if tage[j]["Time.Stamp"][i] >= (auction_tage[j].index - datetime.timedelta(minutes = 5)):
            results_power = pd.read_csv(name, index_col = 't',  sep = ",", error_bad_lines=False)
            results_power=results_power.T
            results_power=results_power[:-1]
            results_power = results_power.astype('float')
            date_list = [auction_tage[j].index[0] + datetime.timedelta(minutes=15*x) for x in range(minute+1, 96)]
            results_power['Date']= date_list
            results_power['Date']=results_power['Date'].apply(lambda x: x.strftime('%H:%M'))
            results_power['Resulting Power [kW] (Old)']=results_power['Power Charge [kW]']+results_power['Power Discharge [kW]']
            #Dataframe mit umgesetztem Fahrplan
            realized_power = pd.concat([fahrplan, soc], axis = 1)
            date_list = [auction_tage[j].index[0] + datetime.timedelta(minutes=15*x) for x in range(0, minute+1)]
            realized_power['Date']= date_list
            realized_power['Date']=realized_power['Date'].apply(lambda x: x.strftime('%H:%M'))
            realized_power = realized_power.rename(columns={0: "Resulting Power [kW]", 1: "Energy [kWh]"})
            results_power = pd.concat([realized_power, results_power], sort=False)
            results_power = results_power.fillna(0)
            
        else:
            #Dataframe with results
            results_power = pd.read_csv(name, index_col = 't',  sep = ",", error_bad_lines=False)
            results_power = results_power.T
            results_power = results_power[:-1]
            results_power = results_power.astype('float')
            date_list = [auction_tage[j].index[0] + datetime.timedelta(minutes=15*x) for x in range(0, 96)]
            results_power['Date']= date_list
            results_power['Date']=results_power['Date'].apply(lambda x: x.strftime('%H:%M'))
            results_power['Resulting Power [kW] (Old)']=results_power['Power Charge [kW]']+results_power['Power Discharge [kW]']
            
        
        #Dataframe with power and energy values
        primary = results_power[['Date','Resulting Power [kW]','Energy [kWh]']].copy()
        primary = primary.rename(columns={"Resulting Power [kW]": "Current schedule [kW]"})
        primary = primary.set_index("Date")
        #Dataframe with prices
        secondary = results_power[['Date','Current Price [EUR/MWh]']].copy()
        secondary = secondary.rename(columns={"Current Price [EUR/MWh]": "Price at time of optimization [EUR/MWh]"})
        #Dataframe with buy- and sell power
        tertiary = results_power[['Date','Power Buy [kW]', 'Power Sell [kW]']].copy()
        tertiary= tertiary.rename(columns={"Power Buy [kW]": "Power purchased [kW]", 
                                           "Power Sell [kW]": "Power sold [kW]"})
        #Plot
        fig, ax1 = plt.subplots()
        ax1 = primary['Current schedule [kW]'].plot(x = 'Date', color="black", lineWidth = 2)
        ax2 = secondary.plot(x = 'Date',color = 'red', secondary_y=True, 
                             ylim =(neg_min_power,max(kap_speicher, max_power)+1), ax=ax1, linestyle = "dashed")
        tertiary.plot(x = 'Date', figsize=(13,6.5),kind='bar',ax=ax1, color =("darkviolet", "pink"),width = 1)
        ax1 = primary['Energy [kWh]'].plot(x = 'Date')
        
        if tage[j]["Time.Stamp"][i] >= (auction_tage[j].index - datetime.timedelta(minutes = 5)):
            minute = int((pd.to_datetime(tage[j]["Time.Stamp"][i]) + datetime.timedelta(minutes = 5)).hour*4
            +(pd.to_datetime(tage[j]["Time.Stamp"][i])+ datetime.timedelta(minutes = 5)).minute/15)
            plt.axvline(x=minute, color ='darkgreen', linewidth=2)
        
        if tage[j]["Time.Stamp"][i][:11]< auction_tage[j].index:
            title = "Optimization " + str(i) + " at trading time D-1 " + tage[j]["Time.Stamp"][i][11:]
        else: 
            title = "Optimization " + str(i) + " at trading time D " + tage[j]["Time.Stamp"][i][11:]
        plt.title(title)
        ax1.set_xlabel('Time [hh:mm]', fontsize = 16)
        ax1.set_ylabel('Power [kW]/ Energy [kWh]', fontsize = 16)
        ax2.set_ylabel('Energy price [€/MWh]', fontsize = 16, color = 'red')
#         ax2.set_ylim(-50,80)
#         ax1.set_ylim(-20,100)
        #Plot von horizontalen Achsen
        ax1.axhline(y = 0, color = "k")
    #     ax2.axhline(y = 0, color = "b")
    
        red_line = mlines.Line2D([], [], color='red', linestyle = "dashed", label='Price at time of optimization [EUR/MWh] (right axis)')
        white_patch = mpatches.Patch(color='white', label = "Profit = " + str(round(-profits[i], 2)) + "€")
        #Legendenplot
        ax1.legend(bbox_to_anchor=(0,1.05,1,0.15), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        ax2.legend(bbox_to_anchor=(0,1.10,1,0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=5, 
                   handles=[red_line, white_patch])
    
        ax2.tick_params(axis='y', colors='red')
        
        n = 12
        for index, label in enumerate(ax1.get_xticklabels()):
            label.set_rotation(90)
            if index % n != 0:
                label.set_visible(False)
        
        #Speicherung von Graphen
        fname = 'figure' + str("%04d" % print_counter)
        plt.savefig(fname)
        plt.close()
        plt.close()
    
revenues[j] = profits[i-1]

