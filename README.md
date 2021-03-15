## Analysing BOVA11 ETF Stock Portfolio and Predicting Daily Changes
Author: Gabriel Tardochi Salles  
Analysis period: May/20 -> Oct/20
### Questions:
- How are different sectors represented on BOVA11 ?
- How did the sectors evolve during this period of time ?
- What was the behavior of stocks within the same sector ? 


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import timedelta
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
import time
%matplotlib inline
plt.style.use('fivethirtyeight')
```

BOVA11 stock portfolio fetched from https://www.blackrock.com/br/products/251816/ishares-ibovespa-fundo-de-ndice-fund..


```python
bova_file = 'BOVA11.xlsx'
bova_stocks = pd.read_excel(bova_file)
bova_stocks.Codigo = bova_stocks.Codigo + '.SA'
bova_stocks.sample(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Codigo</th>
      <th>Nome</th>
      <th>Peso (%)</th>
      <th>Preco</th>
      <th>Cotas</th>
      <th>Valor de mercado</th>
      <th>Valor de face</th>
      <th>Setor</th>
      <th>SEDOL</th>
      <th>Codigo ISIN</th>
      <th>Bolsa</th>
      <th>Localizacao</th>
      <th>Moeda</th>
      <th>Taxa de cambio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>EQTL3.SA</td>
      <td>EQUATORIAL ENERGIA SA</td>
      <td>1.17</td>
      <td>20.87</td>
      <td>6327320</td>
      <td>132051168</td>
      <td>1.320512e+08</td>
      <td>Public Services</td>
      <td>B128R96</td>
      <td>BREQTLACNOR0</td>
      <td>XBSP</td>
      <td>Brasil</td>
      <td>BRL</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>UGPA3.SA</td>
      <td>ULTRAPAR PARTICIPOES SA</td>
      <td>1.14</td>
      <td>18.85</td>
      <td>6815362</td>
      <td>128469574</td>
      <td>1.284696e+08</td>
      <td>Energy</td>
      <td>B0FHTN1</td>
      <td>BRUGPAACNOR8</td>
      <td>XBSP</td>
      <td>Brasil</td>
      <td>BRL</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56</th>
      <td>ELET6.SA</td>
      <td>CENTRAIS ELETR BRAS-ELETROBRAS SER</td>
      <td>0.41</td>
      <td>30.79</td>
      <td>1503206</td>
      <td>46283713</td>
      <td>4.628371e+07</td>
      <td>Public Services</td>
      <td>2308445</td>
      <td>BRELETACNPB7</td>
      <td>XBSP</td>
      <td>Brasil</td>
      <td>BRL</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Consulting Yahoo Finance API to fetch historical data about those stocks, using Alpha Vantage API to fulfill missing data


```python
update_data = False
update_alltime_av_data = False
```


```python
file_name_yf = 'YF_BOVA_FullData.csv'
starting_date = '2020-05-01'
tickers_list = bova_stocks.Codigo.to_list()

# Updates data if wanted(using YF), otherwise continues
if update_data:
    yf_data = yf.download(tickers_list, starting_date)['Close'].reset_index(drop=False)
    yf_data = pd.melt(yf_data, id_vars=['Date'], var_name=['Stock'], value_name='close_price').sort_values(by='Date')
    yf_data.to_csv(file_name_yf, index=False)
    
yf_data = pd.read_csv(file_name_yf)
yf_data.sample(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Stock</th>
      <th>close_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2562</th>
      <td>2020-06-19</td>
      <td>MRVE3.SA</td>
      <td>17.73</td>
    </tr>
    <tr>
      <th>4810</th>
      <td>2020-07-30</td>
      <td>LAME4.SA</td>
      <td>35.02</td>
    </tr>
    <tr>
      <th>4059</th>
      <td>2020-07-16</td>
      <td>ITSA4.SA</td>
      <td>10.15</td>
    </tr>
  </tbody>
</table>
</div>




```python
file_name_av = 'AV_BOVA_FullData.csv'

# Updates data if wanted(using AV), otherwise continues
if update_alltime_av_data:
    with open('alpha_vantage_key.txt') as f:
        ALPHA_VANTAGE_KEY = f.readline()
        f.close()
    ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
    for i in range(len(tickers_list)):
        time.sleep(13)
        this_ticker = tickers_list[i] + 'O'
        av_data, meta = ts.get_daily(this_ticker, outputsize='full')
        av_data['stck'] = tickers_list[i]
        av_data = av_data.loc[:, ['4. close', 'stck']]
        av_data.reset_index(drop=False, inplace=True)
        if i == 0:
            av_total_data = av_data
        else:
            av_total_data = av_total_data.append(av_data, ignore_index = True)
        av_total_data.to_csv(file_name_av, index=False)
        
av_total_data = pd.read_csv(file_name_av)
av_total_data.sample(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>4. close</th>
      <th>stck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>115074</th>
      <td>2007-10-25</td>
      <td>137.65</td>
      <td>CSNA3.SA</td>
    </tr>
    <tr>
      <th>11700</th>
      <td>2017-01-17</td>
      <td>15.82</td>
      <td>PETR4.SA</td>
    </tr>
    <tr>
      <th>174453</th>
      <td>2020-03-03</td>
      <td>30.00</td>
      <td>TAEE11.SA</td>
    </tr>
  </tbody>
</table>
</div>




```python
file_name_price_hist = 'PriceHist_BOVA.csv'
if update_data:
    for i, row in yf_data.iterrows():
        if np.isnan(row['close_price']):
            try:
                yf_data.at[i,'close_price'] = av_total_data[(av_total_data['stck'] == row['Stock']) & (av_total_data['date'] == row['Date'])].iloc[0,1]
            except:
                pass
    yf_data.to_csv(file_name_price_hist, index=False)
    
price_hist = pd.read_csv(file_name_price_hist)
```


```python
# Checking for NaNs..
price_hist[np.isnan(price_hist['close_price'])].reset_index(drop=True).groupby(by='Stock')['Date'].count()
```




    Series([], Name: Date, dtype: int64)



### Sector Performance


```python
# Merging stock information with its history daily price, calculating daily changes
data = price_hist.merge(bova_stocks.loc[:,['Codigo','Nome','Peso (%)','Setor']], left_on='Stock', right_on='Codigo', how='left').drop('Codigo', axis=1)
data['pct_change_d-1'] = data.groupby('Stock')['close_price'].pct_change(fill_method=None).fillna(0)
data['price_norm'] = 0.0
data = data.sort_values(by=['Stock','Date']).reset_index(drop=True)

actual_stock = ''
for i, row in data.iterrows():
    if row['Stock'] != actual_stock:
        data.at[i, 'price_norm'] = 100
        last_price = 100
        actual_stock = row['Stock']
    else:
        this_value = last_price + last_price * abs(row['pct_change_d-1']) if row['pct_change_d-1'] >= 0 else last_price - last_price * abs(row['pct_change_d-1'])
        data.at[i, 'price_norm'] = this_value
        last_price = this_value

print("Stock level dataset:")
display(data.loc[111:,:].head())

# Grouping by sector
gb_setor = data.groupby(['Date','Setor'], as_index=True).agg(Count = ('Setor','count'), Change = ('pct_change_d-1','mean'), PrecoNormalizado = ('price_norm', 'mean'), Share = ('Peso (%)', 'sum')).rename(columns={'Change' : 'Mean_change_d-1'}).reset_index()
print("Grouped by sector:")
display(gb_setor.head())

# Sector Overview
setor_macroview = gb_setor.loc[:,['Setor','Count','Share']].drop_duplicates().sort_values(by='Share', ascending=False).reset_index(drop=True)
print("Sectors overview:")
display(setor_macroview.head())

# Plotting sectors representations on BOVA11 investments
plt.figure(figsize=(16,9))
splot = sns.barplot(x='Setor', y='Count', data=setor_macroview, palette='Greens_d')
splot.set_xticklabels(splot.get_xticklabels(), rotation=45, fontsize=13)
i = 0
for p in splot.patches:
    splot.annotate(str(format(setor_macroview['Share'][i], '.2f'))+'%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()-0.1), 
                   ha = 'center', va = 'center', fontweight='bold', fontsize=12,
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    i += 1

splot.set_title('Investments - BOVA11', fontweight='bold', fontsize=15)
splot.set_xlabel('Sector', fontsize=13, fontweight='bold')
splot.set_ylabel('Number of Companies', fontsize=13, fontweight='bold')

# PLotting sectors evolution over time
plt.figure(figsize=(16,9))
setores = setor_macroview['Setor'].unique()
color_qty = len(setores)
cmap = plt.get_cmap('tab20')
cores = [cmap(i) for i in np.linspace(0, 1, color_qty)]
i = 0
for setor in setores:
    plt.plot(pd.to_datetime(gb_setor[gb_setor['Setor'] == setor]['Date']), gb_setor[gb_setor['Setor'] == setor]['PrecoNormalizado'], color=cores[i])
    i += 1

plt.legend(setores, fontsize=12, fancybox=True, frameon=True, shadow=True)
plt.title('Normalized Average of Sectors Fluctuations', fontsize=15, fontweight='bold')
plt.xlabel('Date', fontsize=13, fontweight='bold')
plt.ylabel('Price (Starting on $100)', fontsize=13, fontweight='bold')
plt.show()
plt.close()

```

    Stock level dataset:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Stock</th>
      <th>close_price</th>
      <th>Nome</th>
      <th>Peso (%)</th>
      <th>Setor</th>
      <th>pct_change_d-1</th>
      <th>price_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>111</th>
      <td>2020-10-08</td>
      <td>ABEV3.SA</td>
      <td>13.68</td>
      <td>AMBEV SA</td>
      <td>3.28</td>
      <td>Consumer Staples</td>
      <td>0.011087</td>
      <td>116.326531</td>
    </tr>
    <tr>
      <th>112</th>
      <td>2020-10-09</td>
      <td>ABEV3.SA</td>
      <td>13.57</td>
      <td>AMBEV SA</td>
      <td>3.28</td>
      <td>Consumer Staples</td>
      <td>-0.008041</td>
      <td>115.391152</td>
    </tr>
    <tr>
      <th>113</th>
      <td>2020-05-04</td>
      <td>AZUL4.SA</td>
      <td>15.16</td>
      <td>AZUL PREF SA</td>
      <td>0.46</td>
      <td>Industrials</td>
      <td>0.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>114</th>
      <td>2020-05-05</td>
      <td>AZUL4.SA</td>
      <td>15.16</td>
      <td>AZUL PREF SA</td>
      <td>0.46</td>
      <td>Industrials</td>
      <td>0.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2020-05-06</td>
      <td>AZUL4.SA</td>
      <td>14.77</td>
      <td>AZUL PREF SA</td>
      <td>0.46</td>
      <td>Industrials</td>
      <td>-0.025726</td>
      <td>97.427445</td>
    </tr>
  </tbody>
</table>
</div>


    Grouped by sector:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Setor</th>
      <th>Count</th>
      <th>Mean_change_d-1</th>
      <th>PrecoNormalizado</th>
      <th>Share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-05-04</td>
      <td>Consumer Discretionary</td>
      <td>13</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-05-04</td>
      <td>Consumer Staples</td>
      <td>9</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>10.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-05-04</td>
      <td>Energy</td>
      <td>5</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>10.77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-05-04</td>
      <td>Financials</td>
      <td>11</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>25.87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-04</td>
      <td>Healthcare</td>
      <td>5</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>4.54</td>
    </tr>
  </tbody>
</table>
</div>


    Sectors overview:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Setor</th>
      <th>Count</th>
      <th>Share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Financials</td>
      <td>11</td>
      <td>25.87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Materials</td>
      <td>9</td>
      <td>17.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Consumer Discretionary</td>
      <td>13</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Energy</td>
      <td>5</td>
      <td>10.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Consumer Staples</td>
      <td>9</td>
      <td>10.62</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_11_6.png)
    



    
![png](output_11_7.png)
    


### Sector Details:
- How did the stocks within an equal sector behave ?
- Wich sectors had a similar behaviour between its companies ?


```python
# Detailment per sector
fig, axes = plt.subplots(nrows=len(setores)//2 + 1, ncols=2, figsize=(17,29), sharey=True)
ax_row = 0
ax_col = 0
for setor in setores:
    this_ax = axes[ax_row][ax_col]
    ax_row = ax_row if ax_col == 0 else ax_row + 1
    ax_col = 0 if ax_col == 1 else 1
    stocks_list = data[data['Setor'] == setor]['Stock'].unique()
    color_qty = len(stocks_list)
    cmap = plt.get_cmap('tab20')
    cores = [cmap(i) for i in np.linspace(0, 1, color_qty)]
    i = 0
    for stock in stocks_list:
        stock_data = data[(data['Setor'] == setor) & (data['Stock'] == stock)]
        this_ax.plot(pd.to_datetime(stock_data['Date']), stock_data['price_norm'], color=cores[i])
        i += 1
    this_legend = [stock[:-3] for stock in stocks_list]
    this_ax.legend(this_legend, fontsize=10, fancybox=True, frameon=True, shadow=True)
    this_ax.set_title(setor, fontsize=15, fontweight='bold')
    #this_ax.set_xlabel('Data', fontsize=13, fontweight='bold')
    if ax_col == 1:
        this_ax.set_ylabel('Price (Starting on $100)', fontsize=13, fontweight='bold')
if len(setores)%2 == 1:
    axes.flat[-1].set_visible(False)
plt.tight_layout()
plt.show()
plt.close()
```


    
![png](output_13_0.png)
    


## Trying to fit a model to predict sector next day price change
### Feature Engineering


```python
# Creating bins with daily price changes based on different ranges(positive and negative changes)
classes = ["b_lvl6","b_lvl5","b_lvl4","b_lvl3","b_lvl2","b_lvl1","neutral","g_lvl1","g_lvl2","g_lvl3","g_lvl4","g_lvl5","g_lvl6"]
def var_lvl(row):
    neutral = 0.001
    lvl1 = 0.005
    lvl2 = 0.01
    lvl3 = 0.02
    lvl4 = 0.03
    lvl5 = 0.04
    change = row['pct_change_d-1'] if 'pct_change_d-1' in row else row['Mean_change_d-1']
    if change >= 0:
        if change <= neutral:
            this_class = "neutral"
        elif change <= lvl1:
            this_class = "g_lvl1"
        elif change <= lvl2:
            this_class = "g_lvl2"        
        elif change <= lvl3:
            this_class = "g_lvl3"
        elif change <= lvl4:
            this_class = "g_lvl4"
        elif change <= lvl5:
            this_class = "g_lvl5"
        else:
            this_class = "g_lvl6"
    else:
        change = abs(change)
        if change <= neutral:
            this_class = "neutral"
        elif change <= lvl1:
            this_class = "b_lvl1"
        elif change <= lvl2:
            this_class = "b_lvl2"        
        elif change <= lvl3:
            this_class = "b_lvl3"
        elif change <= lvl4:
            this_class = "b_lvl4"
        elif change <= lvl5:
            this_class = "b_lvl5"
        else:
            this_class = "b_lvl6"
    return this_class

# Creating the column
gb_setor['set_var_lvl'] = gb_setor.apply(var_lvl, axis=1)
```

#### Simple RandomForestClassifier to predict 'Telecom' performance based on other sectors last 3 days performance


```python
this_setor = 'Telecom'
days = 3
this_df = gb_setor[gb_setor['Setor'] == setor]
this_df = this_df.drop(this_df.columns[1:-1], axis=1).rename(columns={'set_var_lvl': 'output_lvl'}).reset_index(drop=True)
for setor in setores:
    for d in range(1,days+1):
        this_df[setor + ' D-' + str(d)] = ""
    for i,row in this_df.iterrows():
        if i < days:
            continue
        for d in range(1,days+1):
            search_date = this_df.at[i-d,'Date']
            this_df.at[i, setor + ' D-' + str(d)] = gb_setor[(gb_setor['Date'] == search_date) & (gb_setor['Setor'] == setor)]['set_var_lvl'].item()
this_labeled_df = this_df[days:].reset_index(drop=True).drop('Date', axis=1)
display(this_labeled_df.head(3))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output_lvl</th>
      <th>Financials D-1</th>
      <th>Financials D-2</th>
      <th>Financials D-3</th>
      <th>Materials D-1</th>
      <th>Materials D-2</th>
      <th>Materials D-3</th>
      <th>Consumer Discretionary D-1</th>
      <th>Consumer Discretionary D-2</th>
      <th>Consumer Discretionary D-3</th>
      <th>...</th>
      <th>Healthcare D-3</th>
      <th>Telecom D-1</th>
      <th>Telecom D-2</th>
      <th>Telecom D-3</th>
      <th>Technology D-1</th>
      <th>Technology D-2</th>
      <th>Technology D-3</th>
      <th>Real State D-1</th>
      <th>Real State D-2</th>
      <th>Real State D-3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>neutral</td>
      <td>neutral</td>
      <td>g_lvl1</td>
      <td>neutral</td>
      <td>neutral</td>
      <td>g_lvl3</td>
      <td>b_lvl2</td>
      <td>neutral</td>
      <td>...</td>
      <td>neutral</td>
      <td>b_lvl3</td>
      <td>g_lvl4</td>
      <td>neutral</td>
      <td>b_lvl4</td>
      <td>g_lvl1</td>
      <td>neutral</td>
      <td>b_lvl4</td>
      <td>b_lvl4</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>neutral</td>
      <td>g_lvl6</td>
      <td>g_lvl1</td>
      <td>neutral</td>
      <td>b_lvl6</td>
      <td>g_lvl3</td>
      <td>b_lvl2</td>
      <td>...</td>
      <td>b_lvl1</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
      <td>g_lvl1</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
      <td>b_lvl4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b_lvl4</td>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>g_lvl5</td>
      <td>g_lvl6</td>
      <td>g_lvl1</td>
      <td>g_lvl1</td>
      <td>b_lvl6</td>
      <td>g_lvl3</td>
      <td>...</td>
      <td>b_lvl1</td>
      <td>g_lvl3</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>g_lvl5</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34 columns</p>
</div>


Transforming output_lvl to make it a simpler problem (binary classification), making it 1 as a positive price change sign, 0 otherwise


```python
this_labeled_df.output_lvl = this_labeled_df.output_lvl.apply(lambda x: 1 if x.startswith("g") else 0)
this_labeled_df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output_lvl</th>
      <th>Financials D-1</th>
      <th>Financials D-2</th>
      <th>Financials D-3</th>
      <th>Materials D-1</th>
      <th>Materials D-2</th>
      <th>Materials D-3</th>
      <th>Consumer Discretionary D-1</th>
      <th>Consumer Discretionary D-2</th>
      <th>Consumer Discretionary D-3</th>
      <th>...</th>
      <th>Healthcare D-3</th>
      <th>Telecom D-1</th>
      <th>Telecom D-2</th>
      <th>Telecom D-3</th>
      <th>Technology D-1</th>
      <th>Technology D-2</th>
      <th>Technology D-3</th>
      <th>Real State D-1</th>
      <th>Real State D-2</th>
      <th>Real State D-3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>b_lvl3</td>
      <td>neutral</td>
      <td>neutral</td>
      <td>g_lvl1</td>
      <td>neutral</td>
      <td>neutral</td>
      <td>g_lvl3</td>
      <td>b_lvl2</td>
      <td>neutral</td>
      <td>...</td>
      <td>neutral</td>
      <td>b_lvl3</td>
      <td>g_lvl4</td>
      <td>neutral</td>
      <td>b_lvl4</td>
      <td>g_lvl1</td>
      <td>neutral</td>
      <td>b_lvl4</td>
      <td>b_lvl4</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>neutral</td>
      <td>g_lvl6</td>
      <td>g_lvl1</td>
      <td>neutral</td>
      <td>b_lvl6</td>
      <td>g_lvl3</td>
      <td>b_lvl2</td>
      <td>...</td>
      <td>b_lvl1</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
      <td>g_lvl1</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
      <td>b_lvl4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>g_lvl5</td>
      <td>g_lvl6</td>
      <td>g_lvl1</td>
      <td>g_lvl1</td>
      <td>b_lvl6</td>
      <td>g_lvl3</td>
      <td>...</td>
      <td>b_lvl1</td>
      <td>g_lvl3</td>
      <td>b_lvl6</td>
      <td>b_lvl3</td>
      <td>g_lvl5</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
      <td>g_lvl4</td>
      <td>b_lvl6</td>
      <td>b_lvl4</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34 columns</p>
</div>



Simple label encode on the categoricals, basic train/val split


```python
sub = len(classes)//2
for idx, classe in enumerate(classes):
    this_labeled_df = this_labeled_df.replace([classe], idx-sub)

display(this_labeled_df.head(3))
# training on 85% of the data, validating on 15% last datapoints
x_train, x_test, y_train, y_test = train_test_split(this_labeled_df.drop('output_lvl', axis=1, inplace=False),this_labeled_df['output_lvl'], shuffle=False, test_size=0.15)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output_lvl</th>
      <th>Financials D-1</th>
      <th>Financials D-2</th>
      <th>Financials D-3</th>
      <th>Materials D-1</th>
      <th>Materials D-2</th>
      <th>Materials D-3</th>
      <th>Consumer Discretionary D-1</th>
      <th>Consumer Discretionary D-2</th>
      <th>Consumer Discretionary D-3</th>
      <th>...</th>
      <th>Healthcare D-3</th>
      <th>Telecom D-1</th>
      <th>Telecom D-2</th>
      <th>Telecom D-3</th>
      <th>Technology D-1</th>
      <th>Technology D-2</th>
      <th>Technology D-3</th>
      <th>Real State D-1</th>
      <th>Real State D-2</th>
      <th>Real State D-3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>-2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>-3</td>
      <td>4</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-6</td>
      <td>-3</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>-6</td>
      <td>3</td>
      <td>-2</td>
      <td>...</td>
      <td>-1</td>
      <td>-6</td>
      <td>-3</td>
      <td>4</td>
      <td>-6</td>
      <td>-4</td>
      <td>1</td>
      <td>-6</td>
      <td>-4</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4</td>
      <td>-6</td>
      <td>-3</td>
      <td>5</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>-6</td>
      <td>3</td>
      <td>...</td>
      <td>-1</td>
      <td>3</td>
      <td>-6</td>
      <td>-3</td>
      <td>5</td>
      <td>-6</td>
      <td>-4</td>
      <td>4</td>
      <td>-6</td>
      <td>-4</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34 columns</p>
</div>


Building a simple model, predicting and checking its accuracy


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
```


```python
# 100 estimators to make sure model isint overfitting too much (500 -> 82% accuracy)
rf = RandomForestClassifier(n_estimators = 100, random_state=0, class_weight="balanced")
rf.fit(x_train.values, y_train.values)
rf_predict = rf.predict(x_test.values)
```


```python
print(f"Model accuracy on test data: {metrics.accuracy_score(y_test, rf_predict)}")
```

    Model accuracy on test data: 0.7058823529411765
    

##### Even thought this simple RF is achieving a 70% accuracy score, it is probably overfitting here.
##### More data points and a multi folded cross validation(respecting the timeseries aspect of this dataset) would have been welcome to get more confidence on building a model for this task!
##### I wouldnt recommend that you buy/sell stocks based on the above approach, there are many other dimentions that influence on it and that is probably a task for another project :)
