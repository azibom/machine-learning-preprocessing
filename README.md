# machine-learning-preprocessing
Preprocessing in python

## rename
first it is nice to change the name of the columns because we need to call them many times
```python
import pandas as pd
import numpy as np
from sklearn import preprocessing

humans = pd.read_csv('humans.csv')
humans = humans.rename(columns={'first name' : 'name', 'population' : 'pop'})
```

## shape
```python
print(humans.shape)
# result (18, 8)
```

## drop
#### Drop an observation (row)
```python
humans.drop(['row1'], axis = 0, inplace = True)
```

#### Drop a variable (column)
```python
humans.drop(['Age'], axis = 1, inplace = True)
```

## missing value
#### smooth out the null datas
```python
# it replace all of the '0' with NAN
humans.replace(0, np.nan, inplace= True)
```

#### make table for showing you null elements
```python
humans.isnull()
```

#### for showing you null elements shortly
```python
humans.isnull().sum()

# result
# Name              0
# Age               2
# Height            2
# Weight            0
```

## drop he rows that have NAN parametr
```python
humans = humans.dropna(axis=0)
```

## filling NAN
you can use it and consider a desired parameter for missing values
```python
humans = humans.fillna({'Age' : 20})
```

you can use it and consider a previous value for the NAN parametrs 
```python
humans = humans.fillna(method='ffill')
```

you can use it and consider a mean of column for the NAN parametrs 
```python
from sklearn.preprocessing import Imputer
imp = Imputer(missing_value='NaN', strategy='mean', axis=0)
imp.fit(humans)
new_dataset = imp.transform(humans)
```

## duplicates
with this you can omit the repetitive rows 
```python
humans.drop_duplicates()
# or customize it
humans.drop_duplicates(['column1', 'column2'])
```

## concatenating
```python
humand = pd.concat(['human1', 'human2'], axis=0, ignore_index=True)
```

## frequency counts for categorical data
```python
print(humand.Name.value_counts())

# result
# ali              1
# mohammad         4      
# mahsa            7   
# maryam           1 
```

## groupe
you can grope one paranetr and count diffrent things
```python
humans_name = humans.groupby(humans['Name'])
print(humans_name.mean())

#       Age    Height   Weight
# Kate  47.0    169.0     139          
# Luke  34.0    172.0     163          
# Myra  23.0    175       98          
# Neil  36.0    175.0     160          
# Omar  38.0    170.0     145          
# Page  31.0    167.0     135          
# Quin  29.0    171.0     176          
# Ruth  28.0    165.0     131
```

## crosstab
```python
print(pd.crosstab(humans.Name, humans.Age))

# Age   23.0  26.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  36.0  38.0  39.0  41.0  42.0  47.0  53.0
# Name
# Alex     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0
# Bert     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0
# Carl     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0
# Dave     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0
# Fran     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0
# Gwen     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0
# Hank     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0
# Ivan     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1
# Kate     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0
# Luke     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0
# Myra     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
# Neil     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0
# Omar     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0
# Page     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0
# Quin     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0
# Ruth     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0
```

## pivot table
it is one spep more proffesional than crosstab
```python
print(pd.pivot_table(humans, index='Name' ,columns='Age', values='Weight'))

# Age   23.0   26.0   28.0   29.0   30.0   31.0   32.0   33.0   34.0   36.0   38.0   39.0   41.0   42.0   47.0   53.0
# Name
# Alex   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  170.0    NaN    NaN    NaN
# Bert   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  166.0    NaN    NaN
# Carl   NaN    NaN    NaN    NaN    NaN    NaN  155.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Dave   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  167.0    NaN    NaN    NaN    NaN
# Fran   NaN    NaN    NaN    NaN    NaN    NaN    NaN  115.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Gwen   NaN  121.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Hank   NaN    NaN    NaN    NaN  158.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Ivan   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  175.0
# Kate   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  139.0    NaN
# Luke   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  163.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Myra  98.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Neil   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  160.0    NaN    NaN    NaN    NaN    NaN    NaN
# Omar   NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN  145.0    NaN    NaN    NaN    NaN    NaN
# Page   NaN    NaN    NaN    NaN    NaN  135.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Quin   NaN    NaN    NaN  176.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
# Ruth   NaN    NaN  131.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
```

## dumy variables
you can use it for set number for the string datas and now work with them easily
```python
print(pd.get_dummies(humans))
```

## normlize data
you can normat your data with these codes
#### minmax_scale
```python 
from sklearn.preprocessing import scale, normalize, minmax_scale
new_data = minmax_scale(humans, feature_range=(0, 100))
new_data = pd.DataFrame(new_data
                        ,index=humans.index
                        ,columns=humans.columns)
print(new_data)
```

#### normalize
```python 
from sklearn.preprocessing import scale, normalize, minmax_scale
new_data = normalize(humans)
new_data = pd.DataFrame(new_data
                        ,index=humans.index
                        ,columns=humans.columns)
print(new_data)
```

#### scale
```python 
from sklearn.preprocessing import scale, normalize, minmax_scale
new_data = scale(humans)
new_data = pd.DataFrame(new_data
                        ,index=humans.index
                        ,columns=humans.columns)
print(new_data)
```

## outlier
the values that are more than "UX" or less than "LX" are outliers
### Q1 : the number that is more than 25% of values 
### Q3 : the number that is more than 75% of values 
### IQR : Q3 - Q1
### LX : Q1 - (1.5 * IQR)
### UX : Q3 + (1.5 * IQR)

you can count Q1 and Q3 like it
```python 
array = pd.DataFrame(np.array([1, 2, 3, 6, 8, 15 ,55]))
Q1 = array.quantile(0.25)
Q3 = array.quantile(0.25)
```


I hope this data will be useful to you.




