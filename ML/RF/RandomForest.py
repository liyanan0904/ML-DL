import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import load_boston


boston_house = load_boston()
boston_feature_name = boston_house.feature_names
boston_features = boston_house.data 
boston_target = boston_house.target

rgs = RandomForestRegressor(n_estimators=15)
rgs = rgs.fit(boston_features, boston_target)

print rgs.predict(boston_features)


##########---decision tree----#######

from sklearn import tree 
rgs2 = tree.DecisionTreeRegressor() 
rgs2.fit(boston_features, boston_target)

print rgs2.predict(boston_features)
