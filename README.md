# Store Sales TS Forecasting
Our main mission in this competition is, predicting sales for each product family and store combinations.

## Packages

```python
# BASE
# ------------------------------------------------------
import numpy as np
import pandas as pd
import os
import gc
import warnings

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# CONFIGURATIONS
# ------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
```

## Data


There are 6 data that will study on them step by step.

1. Train
2. Test
3. Stores
4. Transactions
5. Daily Oil Price
6. Holiday and Events

**The train data** contains time series of the stores and the product families combination. The sales column gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).The onpromotion column gives the total number of items in a product family that were being promoted at a store at a given date.

**Stores data** gives some information about stores such as city, state, type, cluster.

**Transaction data** is highly correlated with train's sales column. You can understand the sales patterns of the stores.

**Holidays and events data** is a meta data. This data is quite valuable to understand past sales, trend and seasonality components. However, it needs to be arranged. You are going to find a comprehensive data manipulation for this data. That part will be one of the most important chapter in this notebook.

**Daily Oil Price data** is another data which will help us. Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices. That's why, it will help us to understand which product families affected in positive or negative way by oil price.

When you look at the data description, you will see "Additional Notes". These notes may be significant to catch some patterns or anomalies. I'm sharing them with you to remember.

- Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
- A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.# Store-Sales-TS-Forecasting-using-GPU
