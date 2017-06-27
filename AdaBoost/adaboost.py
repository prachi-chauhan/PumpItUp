# AdaBoost Classification
import pandas
import numpy as np
import category_encoders as ce
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

# set all the column names for training_data set
names1 = ['id','amount_tsh','date_recorded','funder','gps_height','installer','longitude','latitude','wpt_name',
          'num_private','basin','subvillage','region','region_code','district_code','lga','ward','population',
          'public_meeting','recorded_by','scheme_management','scheme_name','permit','construction_year',
          'extraction_type','extraction_type_group','extraction_type_class','management','management_group',
          'payment','payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type',
          'source_class','waterpoint_type','waterpoint_type_group']

# set all the column names for labels set
names2 = ['id', 'status_group']

# read the training_data set
df1 = pandas.read_csv('Training_Data.csv', names=names1)

# read the labels set
df2 = pandas.read_csv('Labels.csv', names=names2)

# concatenate training df with labels df to get a single data frame
df = pandas.concat([df1, df2], axis=1)

# drop the column headers
df.drop(df.index[0], inplace=True)

# remove rows with blank cells for some of the features
df['funder'].replace('', np.nan, inplace=True)
df.dropna(subset=['funder'], inplace=True)

df['public_meeting'].replace('', np.nan, inplace=True)
df.dropna(subset=['public_meeting'], inplace=True)

df['scheme_management'].replace('', np.nan, inplace=True)
df.dropna(subset=['scheme_management'], inplace=True)

df['permit'].replace('', np.nan, inplace=True)
df.dropna(subset=['permit'], inplace=True)

# replace the feature 'date_recorded' with a new feature which is equal to (date_recorded - construction_year)
df['date_recorded'].replace('', np.nan, inplace=True)
df.dropna(subset=['date_recorded'], inplace=True)
df['date_recorded'] = pandas.to_datetime(df['date_recorded']).dt.year
df['date_recorded'] = df['date_recorded'].astype('int32')
df['construction_year'] = df['construction_year'].astype('int32')
df['construction_year'] = df['construction_year'].replace(0,np.nan)
df = df.dropna(subset=['construction_year'])
df['date_recorded'] = df['date_recorded']- df['construction_year']

# drop redundant features
df.drop(df.columns[[0,8,9,11,12,13,14,15,16,19,21,23,25,26,28,30,34,36,37,39]], axis=1, inplace=True)

# transform categorical variables to numeric
encoder = ce.OrdinalEncoder(cols=['status_group'])
df = encoder.fit_transform(df)
df = df.apply(pandas.to_numeric, errors='ignore')
encoder = ce.BinaryEncoder()
df = encoder.fit_transform(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Store contents of df into an array
array = df.values
X = array[:,0:67]
Y = array[:,68]

# run Ada Boost algorithm
seed = 7
k = 10
num_trees = 180
kFold = model_selection.KFold(n_splits=k, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kFold)
print(results.mean())
