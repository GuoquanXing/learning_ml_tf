import numpy as np
import pandas as pd
pd.__version__

# pd.Series shall return an array value
city_names = pd.Series(['San Francisco','San Jose','Sacramento'])
population = pd.Series([852469,1015785,485199])
cities = pd.DataFrame({'City Name':city_names,'Population':population})

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe.describe()
california_housing_dataframe.head()

# sample to access DataFrame objects
print('Access Series without header name:')
print(cities['City Name'])
print('Access Series with header name:')
print(cities[['City Name']])
print(cities['City Name'][1])
# print(cities[0:2])

# np.log(population)

# population.apply(lambda val:val > 1000000)

cities['Area Square Miles'] = pd.Series([46.87,176.53,97.92])
cities['Population Density'] = cities['Population'] / cities['Area Square Miles']

# a sample of inner iteration
# print([ city.startswith('San') for city in cities['City Name'] ])

cities['Indicator'] = ( cities['City Name'].apply(lambda name: name.startswith('San')) ) & ( cities['Area Square Miles'] > 50 )
# print(cities)

cities_reindexed = cities.reindex(np.random.permutation(cities.index))
# print(cities_reindexed)
cities_reindex_with_index_na = cities.reindex([3,1,2,5,6])
# print(cities_reindex_with_index_na)