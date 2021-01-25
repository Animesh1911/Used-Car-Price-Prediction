import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TRAIN DATA PREPROCESSING
dataframe_train = pd.read_csv('train-data.csv')

train_data = dataframe_train.copy()

for i in range(0, len(train_data)):
    if train_data['Power'][i] == 'null bhp':
        train_data['Power'][i] = np.nan
        
for i in range(0, len(train_data)):
    if train_data['Mileage'][i] == '0.0 kmpl' or train_data['Mileage'][i] == '0.0 km/kg':
        train_data['Mileage'][i] = np.nan
         
for i in range(0, len(train_data)):
    if train_data['Engine'][i] == 'null CC' or train_data['Engine'][i] == '0 CC':
        train_data['Engine'][i] = np.nan
        
train_data.drop(['New_Price'], axis=1, inplace=True)
train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
train_data.dropna(inplace = True)

train_data.reset_index(inplace = True)

train_data.drop(['index'], axis=1, inplace=True)

y = train_data.iloc[:,-1].values

City = train_data['Location'].unique()

brand=[]

for i in range(0, 5844):
    k = train_data['Name'][i].split()
    brand.append(k[0].upper())
    
Brand = np.array(brand)

fig = plt.figure(figsize=(10,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(Brand)
ax.set_xlabel("Brands")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

Brand = pd.get_dummies(Brand, drop_first = True, dtype=int)

unique_brands=[]
for i in range(0,5844):
    if brand[i] in unique_brands:
        continue
    else:
        unique_brands.append(brand[i])
        
Loc = train_data['Location']

fig = plt.figure(figsize=(10,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(Loc)
ax.set_xlabel("Location")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

Loc = pd.get_dummies(Loc, drop_first = True, dtype=int)

train_data['Seats'] = train_data['Seats'].astype(int)

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Seats'])
ax.set_xlabel("Seats")

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Fuel_Type'])
ax.set_xlabel("Fuel Type")

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Transmission'])
ax.set_xlabel("Transmission")

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Owner_Type'])
ax.set_xlabel("Owner Type")

train_data.replace({'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}, inplace = True)

for i in range(0, 5844):
    k = train_data['Mileage'][i].split()
    train_data['Mileage'][i] = k[0]
    
for i in range(0, 5844):
    k = train_data['Power'][i].split()
    train_data['Power'][i] = k[0]
    
for i in range(0, 5844):
    k = train_data['Engine'][i].split()
    train_data['Engine'][i] = k[0]
    
train_data['Engine'] = train_data['Engine'].astype(int)
train_data['Power'] = train_data['Power'].astype(float)
train_data['Mileage'] = train_data['Mileage'].astype(float)

Fuel = train_data['Fuel_Type']
Fuel = pd.get_dummies(Fuel, drop_first = True, dtype=int)

Trans = train_data['Transmission']
Trans = pd.get_dummies(Trans, drop_first = True, dtype=int)

data_train = pd.concat([train_data, Brand, Loc, Fuel, Trans], axis = 1)

data_train.drop(["Name", "Location", "Fuel_Type",'Transmission','Price'], axis = 1, inplace = True)

#TEST DATA PREPROCESSING
dataframe_test = pd.read_csv('test-data.csv')

test_data = dataframe_test.copy()

for i in range(0, len(test_data)):
    if test_data['Power'][i] == 'null bhp':
        test_data['Power'][i] = np.nan
        
for i in range(0, len(test_data)):
    if test_data['Mileage'][i] == '0.0 kmpl' or test_data['Mileage'][i] == '0.0 km/kg':
        test_data['Mileage'][i] = np.nan
         
for i in range(0, len(test_data)):
    if test_data['Engine'][i] == 'null CC' or test_data['Engine'][i] == '0 CC':
        test_data['Engine'][i] = np.nan
        
test_data.drop(['New_Price'], axis=1, inplace=True)
test_data.drop(['Unnamed: 0'], axis=1, inplace=True)
test_data.dropna(inplace = True)

test_data.reset_index(inplace = True)

test_data.drop(['index'], axis=1, inplace=True)

City_test = test_data['Location'].unique()

brand_test=[]

for i in range(0, 1195):
    k = test_data['Name'][i].split()
    brand_test.append(k[0].upper())
    
Brand_test = np.array(brand_test)

Brand_test = pd.get_dummies(Brand_test, drop_first = True, dtype=int)

unique_brands_test=[]
for i in range(0,1195):
    if brand_test[i] in unique_brands_test:
        continue
    else:
        unique_brands_test.append(brand_test[i])
        
Loc_test = test_data['Location']

Loc_test = pd.get_dummies(Loc_test, drop_first = True, dtype=int)

test_data['Seats'] = test_data['Seats'].astype(int)

test_data.replace({'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}, inplace = True)

for i in range(0, 1195):
    k = test_data['Mileage'][i].split()
    test_data['Mileage'][i] = k[0]
    
for i in range(0, 1195):
    k = test_data['Power'][i].split()
    test_data['Power'][i] = k[0]
    
for i in range(0, 1195):
    k = test_data['Engine'][i].split()
    test_data['Engine'][i] = k[0]
    
test_data['Engine'] = test_data['Engine'].astype(int)
test_data['Power'] = test_data['Power'].astype(float)
test_data['Mileage'] = test_data['Mileage'].astype(float)

Fuel_test = test_data['Fuel_Type']
Fuel_test = pd.get_dummies(Fuel_test, drop_first = True, dtype=int)

Trans_test = test_data['Transmission']
Trans_test = pd.get_dummies(Trans_test, drop_first = True, dtype=int)

data_test = pd.concat([test_data, Brand_test, Loc_test, Fuel_test, Trans_test], axis = 1)

data_test.drop(["Name", "Location", "Fuel_Type",'Transmission'], axis = 1, inplace = True)

# FEATURE SCALING
#Not required for Random Forest Algorithm

#Feature Selection
X = data_train.copy()

plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#Fitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

sns.distplot(y_test-y_pred)

plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

prediction = rf_random.predict(X_test)

plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()

plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

rf_random.score(X_train, y_train)
rf_random.score(X_test, y_test)

#Save Model        
import pickle
file = open('car.pkl', 'wb')

pickle.dump(regressor, file)