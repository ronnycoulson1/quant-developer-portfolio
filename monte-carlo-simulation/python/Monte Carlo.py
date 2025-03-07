#!/usr/bin/env python
# coding: utf-8

# # Simulating Probability in Trading

# # A Monte Carlo Simmulation that predics stock prices based on random prbability moedls

# 1 - Fetch historical data using library yfinance to get historical prices
# 2 - Compute daily returns

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


import yfinance as yf
print("installation working")


# In[3]:


#Fetch msft stock 
stock = yf.Ticker("MSFT")
hist = stock.history(period="1y")
print(hist.tail())


# # Calculate daily return. MSFT over 2020-2025

# In[4]:


get_ipython().system('pip install yahoofinancials')


# In[5]:


import yfinance as yf
#fetch Mricrosoft stock data from Yahoo Finance 
stock = yf.download("MSFT", start = "2020-01-01", end = "2025-01-01")

#print
print(type(stock)) #

#now it has 
stock.columns = stock.columns.get_level_values(0)

#
returns = stock["Close"].pct_change().dropna()
mean = returns.mean()

print("Daily return: ",mean)


# # 2 Calculate Long Returns

# convert stock prices chanegs into loong returns, which are normally distributed and easier to model. Use fomrlua -> ROI = final value - Initial cost / Initial Cost * 100
# 

# In[6]:


import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


#create a DataFRame from Yfinance seaborn

df = yf.download('MSFT', start ='2021-01-01', end = '2023-02-25')
#dsipay first 5 rows of the DataFrame

df.head()


# In[8]:


#SET THE TICKER
stock = yf.Ticker("MSFT")

#FETCH DATA
df = df = stock.history(period ="5d", interval= "1m")

#round off the datefram calues to 2 decimals 
df.round(2)


# In[9]:


df


# In[10]:


#Pt = todays close 
#pt-1 yesterday close 
#rt is the lon return 
#use np.log(df["Close"] / df["Close"].shift(1))
todays_close = df["Close"] #assing a variable to todays close
yesterday_close = df["Close"].shift(1) #assing a variable to yesterdays 

long_term_return = np.log(todays_close / yesterday_close) #calc return 

#dropna values (first row has no previous close)
long_term_return = long_term_return.dropna()


print(long_term_return)


# In[11]:


##creating a histogram 
import matplotlib.pyplot as plt
import numpy as np

plt.hist(long_term_return, bins=30, color = "red", edgecolor = "black")

#adding labels
plt.xlabel("Long Return")
plt.ylabel("Count")
plt.title("Diagram")
plt.show()


# In[12]:


#calculating mean and volatily 
#volatility is calculated using standard deviation or other statistics methods 


import numpy

#use statistics.stdev() 
std_dev_numpy = np.std(long_term_return, ddof = 1) #sample for standar deviation 
print(f"Stock Volatility (Standard Deviation of Log Returns): {std_dev_numpy}")


# # Stcok price movement using Monte Carlo method

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()


# In[14]:


#FETCH TESLA 
#create a DataFRame from Yfinance seaborn

df = yf.download('AMZN', start ='2021-01-01', end = '2023-02-25')
#dsipay first 5 rows of the DataFrame

df.head()


# In[15]:


df = df [["Close"]]
df.head()


# In[16]:


print(df.columns)


# In[17]:


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
    
print(df.columns)


# In[18]:


print(df.head(10))


# In[19]:


# Reset index to move 'Date' into a column
df_reset = df.reset_index()

# Rename 'Price' to 'DATE'
df_reset.rename(columns={"Price": "DATE"}, inplace=True)

# Print updated DataFrame
print(df_reset.head(8))


# In[20]:


df = df_reset


# In[21]:


df.head(5)


# In[22]:


df.rename(columns={"Date": "date", "Close":"price_t"}, inplace=True)
df.head()


# In[25]:


df["price_t-1"] = df["price_t"].shift(1)
df.head()


# In[26]:


df["return_manual"] = (df["price_t"] / df["price_t-1"]) - 1


# In[27]:


df.head()


# In[28]:


df.tail()


# In[31]:


df["return_pct_change_method"] = df["price_t"].pct_change(1)
df.head()


# In[32]:


df["returns"] = (df["price_t"] / df["price_t"].shift(1)) - 1


# In[33]:


df.head()


# In[34]:


df.set_index("date", inplace=True)
df.head()


# In[36]:


df["price_t"].plot(figsize =(12,8))


# In[38]:


df["returns"].plot(figsize=(12, 8))

