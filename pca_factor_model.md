
# Principal Component Analysis As A Factor Model

## Introduction
Principal component analysis (PCA) is a statistical technique which enjoys applications in image processing and quantitative finance.
In this article, we focus on the later application in quantitative trading, in particular using PCA as a multi-factor model of portfolio returns.
We use the multi-factor model to design a momentum trading strategy, backtest it under different investment universes, and present the performance results.

The project is shared on my online repository https://github.com/DinodC/pca-factor-model.

Import packages


```python
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

Magic


```python
%matplotlib inline
```

## Data Collection
In this section, we collect S&P consittuents' historical data from a previous project https://quant-trading.blog/2019/06/24/backtesting-a-trading-strategy-part-2/.

Set S&P Index keys


```python
keys = ['sp500',
        'sp400',
        'sp600']
```

Initialize S&P indices close data


```python
close = {}
```

Pull S&P indices close data


```python
for i in keys: 
    # Load OHLCV data
    with open(i + '_data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    # Update close prices data
    close[i] = data.close.loc['2014-06-12':]
    
    # Close file
    f.close
```

Inspect S&P 500 index constituents data


```python
close['sp500'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>A</th>
      <th>AAL</th>
      <th>AAP</th>
      <th>AAPL</th>
      <th>ABBV</th>
      <th>ABC</th>
      <th>ABMD</th>
      <th>ABT</th>
      <th>ACN</th>
      <th>ADBE</th>
      <th>...</th>
      <th>XEL</th>
      <th>XLNX</th>
      <th>XOM</th>
      <th>XRAY</th>
      <th>XRX</th>
      <th>XYL</th>
      <th>YUM</th>
      <th>ZBH</th>
      <th>ZION</th>
      <th>ZTS</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-06-12</th>
      <td>39.7726</td>
      <td>38.2958</td>
      <td>123.2252</td>
      <td>84.5860</td>
      <td>44.6031</td>
      <td>65.9975</td>
      <td>23.03</td>
      <td>35.7925</td>
      <td>74.5018</td>
      <td>66.56</td>
      <td>...</td>
      <td>25.8547</td>
      <td>41.1019</td>
      <td>83.8928</td>
      <td>46.4230</td>
      <td>28.5372</td>
      <td>35.2780</td>
      <td>51.2950</td>
      <td>101.0367</td>
      <td>27.7535</td>
      <td>30.8448</td>
    </tr>
    <tr>
      <th>2014-06-13</th>
      <td>39.8407</td>
      <td>38.4672</td>
      <td>123.7407</td>
      <td>83.6603</td>
      <td>45.0187</td>
      <td>66.2930</td>
      <td>22.95</td>
      <td>35.7745</td>
      <td>74.7820</td>
      <td>66.82</td>
      <td>...</td>
      <td>25.8884</td>
      <td>41.6091</td>
      <td>84.7098</td>
      <td>46.3744</td>
      <td>28.4920</td>
      <td>36.0532</td>
      <td>51.5945</td>
      <td>101.2861</td>
      <td>27.8192</td>
      <td>30.9702</td>
    </tr>
    <tr>
      <th>2014-06-16</th>
      <td>39.7181</td>
      <td>39.1150</td>
      <td>124.0086</td>
      <td>84.5035</td>
      <td>44.8857</td>
      <td>66.0529</td>
      <td>23.27</td>
      <td>35.8734</td>
      <td>74.8634</td>
      <td>67.62</td>
      <td>...</td>
      <td>26.0740</td>
      <td>41.9028</td>
      <td>84.9326</td>
      <td>46.4424</td>
      <td>28.3791</td>
      <td>35.9318</td>
      <td>51.5099</td>
      <td>100.7105</td>
      <td>27.4060</td>
      <td>30.9798</td>
    </tr>
    <tr>
      <th>2014-06-17</th>
      <td>40.0927</td>
      <td>39.8867</td>
      <td>124.9411</td>
      <td>84.3935</td>
      <td>45.1351</td>
      <td>66.2561</td>
      <td>23.43</td>
      <td>35.8375</td>
      <td>74.5832</td>
      <td>67.54</td>
      <td>...</td>
      <td>26.0910</td>
      <td>42.2409</td>
      <td>84.5200</td>
      <td>46.2677</td>
      <td>28.8310</td>
      <td>36.0346</td>
      <td>51.7639</td>
      <td>100.4707</td>
      <td>27.9601</td>
      <td>31.6355</td>
    </tr>
    <tr>
      <th>2014-06-18</th>
      <td>40.3651</td>
      <td>40.6392</td>
      <td>128.6809</td>
      <td>84.4852</td>
      <td>45.3595</td>
      <td>66.5886</td>
      <td>23.37</td>
      <td>36.2960</td>
      <td>74.7459</td>
      <td>73.08</td>
      <td>...</td>
      <td>26.7810</td>
      <td>42.1964</td>
      <td>84.7758</td>
      <td>46.3841</td>
      <td>28.7406</td>
      <td>36.3521</td>
      <td>51.8876</td>
      <td>101.7178</td>
      <td>27.9977</td>
      <td>31.7126</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 505 columns</p>
</div>



For WordPress


```python
close['sp500'].iloc[:5, :5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>A</th>
      <th>AAL</th>
      <th>AAP</th>
      <th>AAPL</th>
      <th>ABBV</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-06-12</th>
      <td>39.7726</td>
      <td>38.2958</td>
      <td>123.2252</td>
      <td>84.5860</td>
      <td>44.6031</td>
    </tr>
    <tr>
      <th>2014-06-13</th>
      <td>39.8407</td>
      <td>38.4672</td>
      <td>123.7407</td>
      <td>83.6603</td>
      <td>45.0187</td>
    </tr>
    <tr>
      <th>2014-06-16</th>
      <td>39.7181</td>
      <td>39.1150</td>
      <td>124.0086</td>
      <td>84.5035</td>
      <td>44.8857</td>
    </tr>
    <tr>
      <th>2014-06-17</th>
      <td>40.0927</td>
      <td>39.8867</td>
      <td>124.9411</td>
      <td>84.3935</td>
      <td>45.1351</td>
    </tr>
    <tr>
      <th>2014-06-18</th>
      <td>40.3651</td>
      <td>40.6392</td>
      <td>128.6809</td>
      <td>84.4852</td>
      <td>45.3595</td>
    </tr>
  </tbody>
</table>
</div>




```python
close['sp500'].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>A</th>
      <th>AAL</th>
      <th>AAP</th>
      <th>AAPL</th>
      <th>ABBV</th>
      <th>ABC</th>
      <th>ABMD</th>
      <th>ABT</th>
      <th>ACN</th>
      <th>ADBE</th>
      <th>...</th>
      <th>XEL</th>
      <th>XLNX</th>
      <th>XOM</th>
      <th>XRAY</th>
      <th>XRX</th>
      <th>XYL</th>
      <th>YUM</th>
      <th>ZBH</th>
      <th>ZION</th>
      <th>ZTS</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-06-04</th>
      <td>67.95</td>
      <td>29.12</td>
      <td>154.61</td>
      <td>179.64</td>
      <td>76.75</td>
      <td>82.21</td>
      <td>267.06</td>
      <td>77.46</td>
      <td>177.97</td>
      <td>268.71</td>
      <td>...</td>
      <td>57.78</td>
      <td>106.90</td>
      <td>73.59</td>
      <td>54.40</td>
      <td>33.30</td>
      <td>77.40</td>
      <td>106.97</td>
      <td>117.41</td>
      <td>44.40</td>
      <td>108.12</td>
    </tr>
    <tr>
      <th>2019-06-05</th>
      <td>68.35</td>
      <td>30.36</td>
      <td>154.61</td>
      <td>182.54</td>
      <td>77.06</td>
      <td>81.65</td>
      <td>268.80</td>
      <td>78.69</td>
      <td>179.56</td>
      <td>272.86</td>
      <td>...</td>
      <td>59.32</td>
      <td>105.60</td>
      <td>72.98</td>
      <td>55.38</td>
      <td>33.42</td>
      <td>78.88</td>
      <td>107.29</td>
      <td>118.54</td>
      <td>44.18</td>
      <td>108.50</td>
    </tr>
    <tr>
      <th>2019-06-06</th>
      <td>69.16</td>
      <td>30.38</td>
      <td>154.90</td>
      <td>185.22</td>
      <td>77.07</td>
      <td>81.75</td>
      <td>269.19</td>
      <td>80.09</td>
      <td>180.40</td>
      <td>274.80</td>
      <td>...</td>
      <td>59.80</td>
      <td>106.01</td>
      <td>74.31</td>
      <td>55.63</td>
      <td>34.03</td>
      <td>79.15</td>
      <td>108.42</td>
      <td>120.31</td>
      <td>44.24</td>
      <td>108.89</td>
    </tr>
    <tr>
      <th>2019-06-07</th>
      <td>69.52</td>
      <td>30.92</td>
      <td>155.35</td>
      <td>190.15</td>
      <td>77.43</td>
      <td>83.48</td>
      <td>267.87</td>
      <td>80.74</td>
      <td>182.92</td>
      <td>278.16</td>
      <td>...</td>
      <td>59.43</td>
      <td>107.49</td>
      <td>74.58</td>
      <td>55.94</td>
      <td>34.16</td>
      <td>79.56</td>
      <td>109.07</td>
      <td>120.73</td>
      <td>43.64</td>
      <td>110.06</td>
    </tr>
    <tr>
      <th>2019-06-10</th>
      <td>70.29</td>
      <td>30.76</td>
      <td>153.52</td>
      <td>192.58</td>
      <td>76.95</td>
      <td>84.77</td>
      <td>272.43</td>
      <td>81.27</td>
      <td>184.44</td>
      <td>280.34</td>
      <td>...</td>
      <td>59.26</td>
      <td>110.88</td>
      <td>74.91</td>
      <td>57.10</td>
      <td>34.69</td>
      <td>80.38</td>
      <td>108.65</td>
      <td>121.71</td>
      <td>43.84</td>
      <td>110.22</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 505 columns</p>
</div>



For WordPress


```python
close['sp500'].iloc[-6:-1, :5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>A</th>
      <th>AAL</th>
      <th>AAP</th>
      <th>AAPL</th>
      <th>ABBV</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-06-03</th>
      <td>66.99</td>
      <td>27.20</td>
      <td>153.17</td>
      <td>173.30</td>
      <td>75.70</td>
    </tr>
    <tr>
      <th>2019-06-04</th>
      <td>67.95</td>
      <td>29.12</td>
      <td>154.61</td>
      <td>179.64</td>
      <td>76.75</td>
    </tr>
    <tr>
      <th>2019-06-05</th>
      <td>68.35</td>
      <td>30.36</td>
      <td>154.61</td>
      <td>182.54</td>
      <td>77.06</td>
    </tr>
    <tr>
      <th>2019-06-06</th>
      <td>69.16</td>
      <td>30.38</td>
      <td>154.90</td>
      <td>185.22</td>
      <td>77.07</td>
    </tr>
    <tr>
      <th>2019-06-07</th>
      <td>69.52</td>
      <td>30.92</td>
      <td>155.35</td>
      <td>190.15</td>
      <td>77.43</td>
    </tr>
  </tbody>
</table>
</div>




```python
close['sp500'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>A</th>
      <th>AAL</th>
      <th>AAP</th>
      <th>AAPL</th>
      <th>ABBV</th>
      <th>ABC</th>
      <th>ABMD</th>
      <th>ABT</th>
      <th>ACN</th>
      <th>ADBE</th>
      <th>...</th>
      <th>XEL</th>
      <th>XLNX</th>
      <th>XOM</th>
      <th>XRAY</th>
      <th>XRX</th>
      <th>XYL</th>
      <th>YUM</th>
      <th>ZBH</th>
      <th>ZION</th>
      <th>ZTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>...</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.00000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.020384</td>
      <td>41.208459</td>
      <td>145.434801</td>
      <td>135.133436</td>
      <td>66.670458</td>
      <td>84.568308</td>
      <td>162.852343</td>
      <td>49.123858</td>
      <td>118.508010</td>
      <td>142.302737</td>
      <td>...</td>
      <td>39.362328</td>
      <td>58.865112</td>
      <td>75.435801</td>
      <td>53.179431</td>
      <td>26.876008</td>
      <td>51.309177</td>
      <td>66.614527</td>
      <td>111.70591</td>
      <td>36.711273</td>
      <td>59.821177</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.868035</td>
      <td>6.368716</td>
      <td>24.131106</td>
      <td>38.117633</td>
      <td>17.717831</td>
      <td>9.473348</td>
      <td>116.557311</td>
      <td>12.652720</td>
      <td>31.178171</td>
      <td>70.409857</td>
      <td>...</td>
      <td>8.317325</td>
      <td>22.243159</td>
      <td>4.713198</td>
      <td>7.834761</td>
      <td>3.222998</td>
      <td>16.350151</td>
      <td>15.998532</td>
      <td>9.92955</td>
      <td>10.738373</td>
      <td>20.214756</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.258600</td>
      <td>24.539800</td>
      <td>79.168700</td>
      <td>82.743800</td>
      <td>42.066600</td>
      <td>65.718100</td>
      <td>22.220000</td>
      <td>33.935700</td>
      <td>68.852300</td>
      <td>60.880000</td>
      <td>...</td>
      <td>25.247700</td>
      <td>32.502600</td>
      <td>58.967500</td>
      <td>34.178400</td>
      <td>18.532600</td>
      <td>28.874500</td>
      <td>44.181800</td>
      <td>89.23610</td>
      <td>18.885300</td>
      <td>30.652000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.391500</td>
      <td>36.574400</td>
      <td>132.526400</td>
      <td>103.637500</td>
      <td>53.175300</td>
      <td>77.745400</td>
      <td>80.700000</td>
      <td>39.410700</td>
      <td>92.630500</td>
      <td>82.100000</td>
      <td>...</td>
      <td>31.489100</td>
      <td>42.116300</td>
      <td>72.522500</td>
      <td>48.734500</td>
      <td>24.214700</td>
      <td>35.032500</td>
      <td>53.061000</td>
      <td>103.35200</td>
      <td>26.823800</td>
      <td>44.771300</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>46.123400</td>
      <td>40.847900</td>
      <td>150.479700</td>
      <td>119.476200</td>
      <td>58.475100</td>
      <td>83.862300</td>
      <td>118.580000</td>
      <td>43.045400</td>
      <td>112.149800</td>
      <td>107.970000</td>
      <td>...</td>
      <td>39.059500</td>
      <td>52.731200</td>
      <td>75.815100</td>
      <td>54.533600</td>
      <td>26.571900</td>
      <td>48.112000</td>
      <td>61.584700</td>
      <td>112.77000</td>
      <td>38.252800</td>
      <td>51.338500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>65.603800</td>
      <td>46.127100</td>
      <td>161.921100</td>
      <td>167.857700</td>
      <td>82.898700</td>
      <td>89.855500</td>
      <td>261.940000</td>
      <td>58.174900</td>
      <td>149.608800</td>
      <td>212.280000</td>
      <td>...</td>
      <td>45.463000</td>
      <td>68.820200</td>
      <td>78.488900</td>
      <td>59.653000</td>
      <td>29.655300</td>
      <td>67.341800</td>
      <td>80.207700</td>
      <td>119.13230</td>
      <td>46.435300</td>
      <td>80.866200</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81.940000</td>
      <td>57.586600</td>
      <td>199.159900</td>
      <td>229.392000</td>
      <td>116.445400</td>
      <td>107.649700</td>
      <td>449.750000</td>
      <td>81.270000</td>
      <td>184.440000</td>
      <td>289.250000</td>
      <td>...</td>
      <td>59.800000</td>
      <td>139.263300</td>
      <td>86.137400</td>
      <td>67.795300</td>
      <td>35.000000</td>
      <td>83.549000</td>
      <td>109.070000</td>
      <td>130.91280</td>
      <td>57.139500</td>
      <td>110.220000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 505 columns</p>
</div>



For WordPress


```python
close['sp500'].describe().iloc[:, :5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>A</th>
      <th>AAL</th>
      <th>AAP</th>
      <th>AAPL</th>
      <th>ABBV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
      <td>1257.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.020384</td>
      <td>41.208459</td>
      <td>145.434801</td>
      <td>135.133436</td>
      <td>66.670458</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.868035</td>
      <td>6.368716</td>
      <td>24.131106</td>
      <td>38.117633</td>
      <td>17.717831</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.258600</td>
      <td>24.539800</td>
      <td>79.168700</td>
      <td>82.743800</td>
      <td>42.066600</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.391500</td>
      <td>36.574400</td>
      <td>132.526400</td>
      <td>103.637500</td>
      <td>53.175300</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>46.123400</td>
      <td>40.847900</td>
      <td>150.479700</td>
      <td>119.476200</td>
      <td>58.475100</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>65.603800</td>
      <td>46.127100</td>
      <td>161.921100</td>
      <td>167.857700</td>
      <td>82.898700</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81.940000</td>
      <td>57.586600</td>
      <td>199.159900</td>
      <td>229.392000</td>
      <td>116.445400</td>
    </tr>
  </tbody>
</table>
</div>




```python
close['sp500'].shape
```




    (1257, 505)



Fill NaNs with previous observation


```python
close['sp500'].fillna(method='ffill', inplace=True)
close['sp400'].fillna(method='ffill', inplace=True)
close['sp600'].fillna(method='ffill', inplace=True)
```

Initialize daily returns


```python
returns = {}
```

Calculate daily returns


```python
for i in keys:
    returns[i] = close[i].pct_change()
```

## Momentum Strategy Implementation
In this section, we present the model (factor model) and tool (principal compenent analysis) used in implementing the momentum trading strategy. 

### 1. Factor Models
Factor models use economic (e.g. interest rates), fundamental (e.g. price per earnings), and statistical (e.g. principal component analysis) factors to explain asset prices (and returns).
Fama and French initially designed the [three-factor model](https://en.wikipedia.org/wiki/Fama–French_three-factor_model) which extends the capital asset pricing model (CAPM) to include size and value factors.
The general framework is known as the [arbitrage pricing theory (APT)](https://en.wikipedia.org/wiki/Arbitrage_pricing_theory) developed by Stephen Ross and proposes multiple factors. 

### 2. Principal Component Analysis (PCA)
Principal component analysis is a statistical procedure for finding patterns in high dimension data.
PCA allows you to compress high dimension data by reducing the number of dimensions, without losing much information.
Principal component analysis has the following applications in quantitative finance: interest-rate modeling and portfolio analysis.

PCA implementation has the following steps:
1. Pull data
2. Adjust data by subtracting the mean
3. Calculate the covariance matrix of the adjusted data
4. Calculate the eigenvectors and eigenvalues of the covariance matrix
5. Choose the most significant components which explain around 95% of the data

Note that in this article we are only interested in applying principal component analysis, for a complete illustration you can start by checking out https://en.wikipedia.org/wiki/Principal_component_analysis#Applications.

### 3. Momentum Trading Strategy
Momentum trading strategies profit from current market trends continuation.
The momentum trading strategy proposed below assumes that factor returns have momentum.
The idea is to long the *winning* (*losing*) stocks which have the highest (lowest) expected returns according to factors.

Before implementation, we set the following parameters:
1. Lookback period
2. Number of significant factors
3. Number of winning and losing stocks to pick 


```python
lookback = 250
number_of_factors = 5
top_n = 50
```

Initialize the trading positions


```python
positions = {}

for i in keys:
    # Update positions
    positions[i] = pd.DataFrame(np.zeros((returns[i].shape[0], returns[i].shape[1])),
                                 index=returns[i].index,
                                 columns=returns[i].columns
                                )
```

Implementation


```python
for i in keys:
    for j in range(lookback + 1, len(close[i])):
        # Calculate the daily returns
        R = returns[i].iloc[j - lookback + 1:j, :]

        # Avoid daily returns with NaNs
        has_data = (R.count() == max(R.count()))
        has_data_list = list(R.columns[has_data])
        R = R.loc[:, has_data_list]

        # Calculate the mean of the daily returns
        R_mean = R.mean()

        # Calculate the adjusted daily returns
        R_adj = R.sub(R_mean)

        # Calculate the covariance matrix
        cov = R_adj.cov()

        # Calculate the eigenvalues (B) and eigenvectors (X)
        eigen = np.linalg.eig(cov)
        B = eigen[0]
        X = eigen[1]

        # Retain only a number of factors
        X = X[:, :number_of_factors]

        # OLS
        model = sm.OLS(R_adj.iloc[-1], X)
        results = model.fit()
        b = results.params

        # Calculate the expected returns
        R_exp = R_mean.add(np.matmul(X, b))

        # Momentum strategy
        shorts = R_exp.sort_values()[:top_n].index
        positions[i].iloc[j][shorts] = -1
        longs = R_exp.sort_values()[-top_n:].index
        positions[i].iloc[j][longs] = 1
```

Remarks:
1. Investment universes used in backtesting are the S&P 500, S&P 400 MidCap and S&P 600 SmallCap indices.
2. Ordinary least squares (OLS) method is used to calculate stocks' expected returns from significant factors.
3. Only a single stock is bought for each of the winning (losing) stocks, this could be improved by adjusting the number by the rank. 

## Performance Analysis
In this section, we present the performance of the momentum trading strategy based on principal component analysis.

Adjust the positions because we consider close prices


```python
for i in keys:
    positions[i] = positions[i].shift(periods=1)
```

Calculate the daily PnL of the momentum strategy


```python
pnl_strat = {}
avg_pnl_strat = {}

for i in keys:
    # Daily pnl
    pnl_strat[i] = (positions[i].mul(returns[i])).sum(axis='columns')
    # Annualized average pnl of the momentum strategy
    avg_pnl_strat[i] = pnl_strat[i].mean() * 250
```

Average daily PnL of momentum strategy using different investment universes i.e. S&P 500, S&P 400 and S&P 600 indices


```python
pd.DataFrame(avg_pnl_strat,
            index=['Average PnL'],
            )
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sp500</th>
      <th>sp400</th>
      <th>sp600</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Average PnL</th>
      <td>-0.033961</td>
      <td>-0.491665</td>
      <td>-1.941058</td>
    </tr>
  </tbody>
</table>
</div>



Remark: the average daily pnl of the momentum strategy is negative regardless of the investment universe used.

Plot the cumulative PnL of the momentum trading strategy


```python
# Set size
plt.figure(figsize=(20, 10))

for i in range(len(keys)):
    plt.plot(pnl_strat[keys[i]].cumsum())
    
plt.xlim(pnl_strat['sp500'].index[0], pnl_strat['sp500'].index[-1])
# plt.ylim(-10, 5)

# Set title and legend
plt.title('Cumulative PnL Of Momentum Trading Strategy')
plt.legend(keys)
```




    <matplotlib.legend.Legend at 0x1c192b4978>




![png](output_47_1.png)


Remarks: 
1. From July 2015 to January 2017, the momentum trading strategy generated negative PnL.
2. From July 2017 to January 2019, the momentum trading strategy turned it around and generated positive PnL.
3. From January 2019, the momentum trading strategy continuted to generate positive PnL for S&P 500 and S&P 400 MidCap indices only. 

## Conclusion
In this article, we implemented a momentum trading strategy based on principal component analysis.
The momentum trading strategy generated positive PnL from July 2017 to January 2019, and negative PnL from July 2015 to July 2017.
A way to enhance the current momentum trading strategy is to include exit and entry points depending on the expected profitability of the trading system.
