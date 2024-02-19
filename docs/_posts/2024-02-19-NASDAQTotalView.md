---
layout: post 
title: "NASDAQ Total View" 
date: 2024-02-19
--- 

```python 

%pip install tqdm
%pip install pytables

from pathlib import Path

from tqdm import tqdm 

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

```
```python 
nasdaq_path = Path('nasdaq100')
# list(nasdaq_path.iterdir())
```


```python 
tcols = ['openbartime',
         'firsttradetime',
         'highbidtime',
         'highasktime',
         'hightradetime',
         'lowbidtime',
         'lowasktime',
         'lowtradetime',
         'closebartime',
         'lasttradetime']

drop_cols = ['unknowntickvolume',
             'cancelsize',
             'tradeatcrossorlocked']

columns = {'volumeweightprice': 'price',
           'finravolume': 'fvolume',
           'finravolumeweightprice': 'fprice',
           'uptickvolume': 'up',
           'downtickvolume': 'down',
           'repeatuptickvolume': 'rup',
           'repeatdowntickvolume': 'rdown',
           'firsttradeprice': 'first',
           'hightradeprice': 'high',
           'lowtradeprice': 'low',
           'lasttradeprice': 'last',
           'nbboquotecount': 'nbbo',
           'totaltrades': 'ntrades',
           'openbidprice': 'obprice',
           'openbidsize': 'obsize',
           'openaskprice': 'oaprice',
           'openasksize': 'oasize',
           'highbidprice': 'hbprice',
           'highbidsize': 'hbsize',
           'highaskprice': 'haprice',
           'highasksize': 'hasize',
           'lowbidprice': 'lbprice',
           'lowbidsize': 'lbsize',
           'lowaskprice': 'laprice',
           'lowasksize': 'lasize',
           'closebidprice': 'cbprice',
           'closebidsize': 'cbsize',
           'closeaskprice': 'caprice',
           'closeasksize': 'casize',
           'firsttradesize': 'firstsize',
           'hightradesize': 'highsize',
           'lowtradesize': 'lowsize',
           'lasttradesize': 'lastsize',
           'tradetomidvolweight': 'volweight',
           'tradetomidvolweightrelative': 'volweightrel'}
```
```python 
path = nasdaq_path/ '1min_taq'
if not path.exists():
    path.mkdir(parents=True)
    
data = [] 
    
# read files and create progress bar
for f in tqdm(list(nasdaq_path.glob('*/**/*.csv.gz'))):
    data.append(pd.read_csv(f, parse_dates=[['Date', 'TimeBarStart']])
                    .rename(columns=str.lower)
                    .drop(tcols + drop_cols, axis=1)
                    .rename(columns=columns)
                    .set_index('date_timebarstart')
                    .sort_index()
                    .between_time('9:30', '16:00')
                    .set_index('ticker', append=True)
                    .swaplevel()
                    .rename(columns=lambda x: x.replace('tradeat', 'at')))
    
data = pd.concat(data).apply(pd.to_numeric, downcast='integer')
data.index.rename(['ticker', 'date_time'], inplace=True)
print(data.info(show_counts=True))
data.to_hdf(nasdaq_path / 'algoseek.h5', 'min_taq')
```
```python 
data.info(null_counts = True)
```
```python 
len(data.index.unique('ticker'))
```
```python 
constituents = (data.groupby([data.index.get_level_values('date_time').date, 'ticker'])
                .size()
                .unstack('ticker')
                .notnull()
                .astype(int)
                .replace(0, np.nan))
```
```python 
constituents.index = pd.to_datetime(constituents.index)
constituents = constituents.resample('M').max()
constituents.index = constituents.index.date
```

```python 
fig, ax = plt.subplots(figsize=(20, 20))
mask = constituents.T.isnull()
ax = sns.heatmap(constituents.T, mask=mask, cbar=False, ax=ax, cmap='Blues_r')
ax.set_ylabel('')
fig.suptitle('NASDAQ100 Constituents (2015)')
fig.tight_layout();
```

![image]({{site.url}}/assets/images/NASDAQTOtalView_files/myimg.png){: width="3000" } 
