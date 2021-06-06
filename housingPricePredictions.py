#For this code to run, KaggleSubmissionFile10836.csv is to be submitted as input file.
#The concept here is the best score file is fedback to the model as training data along with actual train.csv


import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import numpy as ndarray
import xgboost as xgb
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.kernel_ridge import KernelRidge
#from sklearn.cross_validation import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import norm
#from scipy.weave import inline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition.pca import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import decomposition, ensemble
from sklearn import utils
from sklearn import preprocessing
from sklearn import tree
from scipy import stats
from sklearn import neighbors
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

TARGET = 'SalePrice'
NFOLDS = 4
SEED = 0
NROWS = None
# SUBMISSION_FILE = '../input/sample_submission.csv'
SUBMISSION_FILE = 'C:/Users/Ameema Zainab/submissions.csv'

train = pd.read_csv("C:/Users/Ameema Zainab/Desktop/kaggle/train.csv")
train= train[train["GrLivArea"] < 4000]
test_read = pd.read_csv("C:/Users/Ameema Zainab/Desktop/kaggle/test.csv")




#visualisations
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

plt.show()

####################################################FINDING THE ORDER OF IMPORTANCE OF THE VARIABLES ON SALE PRICE####################################################
corr=train.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]

#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()

#reading in the test predictions set from the best score
test_pred = pd.read_csv("D:\KaggleSubmissionFile10836.csv")
#importing the best score to be compared with the results
y_final_read = pd.read_csv("D:\KaggleSubmissionFile10836.csv")
y_final = y_final_read["SalePrice"]
#combined dataset
test_read['SalePrice'] = test_pred["SalePrice"]

test_read.loc[test_read['MSSubClass'] == 20, 'MSZoning'] = 'RL'
test_read.loc[test_read['MSSubClass'] == 30, 'MSZoning'] = 'RM'
test_read.loc[test_read['MSSubClass'] == 70, 'MSZoning'] = 'RM'
test_read.loc[test_read['SaleType'].isnull(), 'SaleType'] = 'WD'
test_read.loc[test_read['BsmtFinType1'].isnull(), 'TotalBsmtSF'] = 0
test_read.loc[666, "GarageQual"] = "TA"
test_read.loc[666, "GarageCond"] = "TA"
test_read.loc[666, "GarageFinish"] = "Unf"
test_read.loc[666, "GarageYrBlt"] = "1980"
test_read.loc[1116, "GarageType"] = np.nan

X = pd.concat( (train.loc[:,'MSSubClass':'SalePrice'],test_read.loc[:,'MSSubClass':'SalePrice']) ,ignore_index=True)
predictors = X.loc[:,'MSSubClass':'SaleCondition']


# Looking at categorical values
def cat_exploration(column):return X[column].value_counts()
#Imputing the missing values
def cat_imputation(column, value):X.loc[X[column].isnull(),column] = value
#missing data #train dataset # test dataset
def show_missing():
    missing = X.columns[X.isnull().any()].tolist()
    return missing


# X["LotFrontage"] = X.groupby("Neighborhood").transform(lambda x: x.fillna(x.mean()))
X.loc[X.LotFrontage.isnull(), 'LotFrontage'] = X.groupby('Neighborhood').LotFrontage.transform('median')

X["CentralAir"] = (X["CentralAir"] == "Y") * 1.0

X.loc[X['Alley'].isnull(), 'Alley'] = 'No alley access'
X.loc[:, "KitchenQual"] = X.loc[:, "KitchenQual"].fillna("TA")

del X['1stFlrSF']
del X['2ndFlrSF']
del X['LowQualFinSF']
# only three values not matching for train and many for test
# del X['Neighborhood']
# Seem to have similar values related to MasVnrType
del X['MasVnrType']
del X['BsmtFullBath']
del X['BsmtHalfBath']
del X['HalfBath']
del X['Foundation']
del X['MSSubClass']
del X['MSZoning']

cat_imputation('MasVnrArea', 'None')
cat_imputation('PoolQC','None')
cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')
cat_imputation('Electrical','SBrkr')
cat_imputation('FireplaceQu','None')

X["Is_Electrical_SBrkr"] = (X["Electrical"] == "SBrkr") * 1

basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
X[basement_cols][X['BsmtQual'].isnull()==True]
for cols in basement_cols:
    if 'FinSF'not in cols: cat_imputation(cols,'None')

del X['BsmtFinSF1']
del X['BsmtFinSF2']
del X['BsmtUnfSF']

garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
X[garage_cols][X['GarageType'].isnull()==True]
for cols in garage_cols:
    if X[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)

X['HasBsmt'] = pd.Series(len(X['TotalBsmtSF']), index=X.index)
X['HasBsmt'] = 0
X.loc[X['TotalBsmtSF']>0,'HasBsmt'] = 1
X.loc[X['HasBsmt']==1,'TotalBsmtSF'] = np.log(X['TotalBsmtSF'])

quality_dict = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
X["ExterQual"] = X["ExterQual"].map(quality_dict).astype(int)
X["ExterCond"] = X["ExterCond"].map(quality_dict).astype(int)
X["BsmtQual"] = X["BsmtQual"].map(quality_dict).astype(int)
X["BsmtCond"] = X["BsmtCond"].map(quality_dict).astype(int)
X["HeatingQC"] = X["HeatingQC"].map(quality_dict).astype(int)
X["KitchenQual"] = X["KitchenQual"].map(quality_dict).astype(int)
X["FireplaceQu"] = X["FireplaceQu"].map(quality_dict).astype(int)
X["GarageQual"] = X["GarageQual"].map(quality_dict).astype(int)
X["GarageCond"] = X["GarageCond"].map(quality_dict).astype(int)

# cat_imputation('LotFrontage',(X['LotArea']/2)**(.5))

X[show_missing()].isnull().sum()

#applying log transformation
X['SalePrice'] = np.log(X['SalePrice'])

#data transformation
X['GrLivArea'] = np.log(X['GrLivArea'])

X["Aggregate_OverallQual"] = X.OverallQual.replace( {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})

X["New_House"] = (X["YearRemodAdd"] == X["YrSold"]) * 1


#convert categorical variable into dummy
#good-way to convert categorical variable into dummy without increasing the column numbers
# X = X.replace({
#     "MSZoning":{"A":1,"C (all)":2, "FV":3, "I":4, "RH":5, "RL":6, "RP":7, "RM":8},
#     "Street":{"Grvl":1,"Pave":2},
#     "LotShape":{"Reg":1,"IR1":2, "IR2":3, "IR3":4},
#     "LandContour":{"Lvl":1,"Bnk":2, "HLS":3, "Low":4},
#     "Utilities":{"AllPub":1,"NoSewr":2, "NoSeWa":3, "ELO":4},
#     "LotConfig":{"Inside":1,"Corner":2, "CulDSac":3, "FR2":4, "FR3":5},
#     "LandSlope":{"Gtl":1,"Mod":2, "Sev":3},
#     "Neighborhood":{"Blmngtn":1, "Blueste":2, "BrDale":3, "BrkSide":4, "ClearCr":5, "CollgCr":6, "Crawfor":7, "Edwards":8, "Gilbert":9, "IDOTRR":10, "MeadowV":11, "Mitchel":12, "Names":13, "NoRidge":14, "NPkVill":15, "NridgHt":16, "NWAmes":17, "OldTown":18, "SWISU":19, "Sawyer":20, "SawyerW":21, "Somerst":22, "StoneBr":23, "Timber":24, "Veenker":25},
#     "Condition1":{"Artery":1, "Feedr":2, "Norm":3, "RRNn":4, "RRAn":5, "PosN":6, "PosA":7, "RRNe":8, "RRAe":9},
#     "Condition2":{"Artery":1, "Feedr":2, "Norm":3, "RRNn":4, "RRAn":5, "PosN":6, "PosA":7, "RRNe":8, "RRAe":9},
#     "BldgType":{"1Fam":1, "2FmCon":2, "Duplx":3, "TwnhsE":4, "TwnhsI":5},
#     "HouseStyle":{"1Story":1, "1.5Fin":2, "1.5Unf":3, "2Story":4, "2.5Fin":5, "2.5Unf":6, "SFoyer":7, "SLvl":8},
#     "RoofStyle":{"Flat":1, "Gable":2, "Gambrel":3, "Hip":4, "Mansard":5, "Shed":6},
#     "RoofMatl":{"ClyTile":1, "CompShg":2, "Membran":3, "Metal":4, "Roll":5, "Tar&Grv":6, "WdShake":7, "WdShngl":8},
#     "Exterior1st":{ "AsbShng":1,  "AsphShn":2,  "BrkComm":3,  "BrkFace":4,  "CBlock":5,  "CemntBd":6,  "HdBoard":7,  "ImStucc":8,  "MetalSd":9,  "Other	":10,  "Plywood":11,  "PreCast":12,  "Stone":13,  "Stucco":14,  "VinylSd":15,  "Wd Sdng":16,  "WdShing":17},
#     "Exterior2nd":{ "AsbShng":1,  "AsphShn":2,  "BrkComm":3,  "BrkFace":4,  "CBlock":5,  "CemntBd":6,  "HdBoard":7,  "ImStucc":8,  "MetalSd":9,  "Other	":10,  "Plywood":11,  "PreCast":12,  "Stone":13,  "Stucco":14,  "VinylSd":15,  "Wd Sdng":16,  "WdShing":17},
#     "MasVnrType":{"BrkCmn":1, "BrkFace":2, "CBlock":3, "None":4, "Stone":5},
#     "ExterQual":{"Ex":1, "Gd":2, "TA":3, "Fa":4, "Po":5},
#     "ExterCond":{"Ex":1, "Gd":2, "TA":3, "Fa":4, "Po":5},
#     "Foundation":{"BrkTil":1, "CBlock":2, "PConc":3, "Slab":4, "Stone":5, "Wood":6},
#     "BsmtQual":{"Ex":1, "Gd":2, "TA":3, "Fa":4, "Po":5, "No":6},
#     "BsmtCond":{"Ex":1, "Gd":2, "TA":3, "Fa":4, "Po":5, "No":6},
#     "BsmtExposure":{"Gd":5, "Av":4, "Mn":3, "No":2, "NoBasement":1},
#     "BsmtFinType1":{"NoBasement":1, "Unf":2, "LwQ":3, "Rec":4, "BLQ":5, "ALQ":6, "GLQ":7},
#     "BsmtFinType2":{"NoBasement":1, "Unf":2, "LwQ":3, "Rec":4, "BLQ":5, "ALQ":6, "GLQ":7},
#     "SaleType":{ "WD ":1,  "CWD":2,  "VWD":3,  "New":4,  "COD":5,  "Con":6,  "ConLw":7,  "ConLI":8,  "ConLD":9,  "Oth":10},
#     "SaleCondition":{ "Normal":1,  "Abnorml":2,  "AdjLand":3,  "Alloca":4,  "Family":5,  "Partial":6},
#     "Heating":{"Floor":1, "GasA":2, "GasW":3, "Grav":4, "OthW":5, "Wall":6},
#     "HeatingQC":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1},
#     "CentralAir":{"N":2, "Y":1},
#     "Electrical":{"SBrkr":5, "FuseA":4, "FuseF":3, "FuseP":2, "Mix":1},
#     "KitchenQual":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1},
#     "Functional":{ "Typ":1,  "Min1":2,  "Min2":3,  "Mod":4,  "Maj1":5,  "Maj2":6,  "Sev":7,  "Sal":8},
#     "GarageType":{ "2Types":1,  "Attchd":2,  "Basment":3,  "BuiltIn":4,  "CarPort":5,  "Detchd":6,  "No":7},
#     "GarageFinish":{"Fin":4, "RFn":3, "Unf":2, "No":1},
#     "GarageQual":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, "No":6},
#     "GarageCond":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, "No":6},
#     "PavedDrive":{"Y":5, "P":4, "N":3}
#  })
X = pd.get_dummies(X)

X= X.fillna(X.mean())

X_train = X[:len(train)]
X_test=X[len(train):]
Y = X['SalePrice']

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train_part, X_val, Y_train_part, Y_val = train_test_split(X, Y)

X_test = pd.get_dummies(X_test)
X_test= X_test.fillna(X_test.mean())

# Regressors and Classifiers

#Applying PCA
pca = PCA(n_components=150).fit(X)
pc = pca.components_
X_pca_train = pca.transform(X)
X_pca_test = pca.transform(X_test)


# # param_test2 = {'max_depth':range(1,10,2)}
# # params = GridSearchCV(estimator = GradientBoostingRegressor( n_estimators= 500,
# #         learning_rate= 0.2, loss= 'huber',alpha=0.95),
# # param_grid = param_test2, n_jobs=4,iid=False, cv=5)
# # params.fit(X_pca_train,Y)
# # params.grid_scores_, params.best_params_, params.best_score_
# 
# params = {'n_estimators': 400, 'max_depth': 10,
#         'learning_rate': 0.35, 'loss': 'huber','alpha':0.99}
# GBM = ensemble.GradientBoostingRegressor(**params).fit(X_pca_train, Y)
# gbm_preds = np.exp(GBM.predict(X_pca_test))
# 
# et_regr = ExtraTreesRegressor()
# et_regr.fit(X, Y)
# # Run prediction on training set to get a rough idea of how well it does.
# et_regr_preds = np.exp(et_regr.predict(X_test))
# 
# thresholds = np.sort(et_regr.feature_importances_)
# for thresh in thresholds:
#     selection = SelectFromModel(et_regr, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X)
#     selection_model = ExtraTreesRegressor()
#     selection_model.fit(select_X_train, Y)
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     print("Thresh=%.3f, n=%d, RMSE= %.10f" % (thresh, select_X_train.shape[1], np.sqrt(mean_squared_error(np.log(y_final), np.log(y_pred)))))
# 
# selection = SelectFromModel(et_regr, threshold=thresh, prefit=True)
# select_X_train = selection.transform(X)
# # train model
# selection_model = ExtraTreesRegressor()
# selection_model.fit(select_X_train, Y)
# # eval model
# select_X_test = selection.transform(X_test)
# et_regr_preds = np.exp(selection_model.predict(select_X_test))
# 
# 
# #preds = 0*lasso_preds + 0*xgb_preds + 0*kNN_preds +0*RF_preds + 0*ENCV_preds + 0*xgb_pca_preds + 1*gbm_preds
# preds = 0.5*gbm_preds +0.5*et_regr_preds
# 
# preds1 = np.round(preds)
# solution = pd.DataFrame({"id":test_read.Id, "SalePrice":preds1})
# solution.to_csv("ridge_sol.csv", index = False)
# 
# #check with best score
# mse = mean_squared_error(np.log(y_final), np.log(preds1))
# rmse_final=np.sqrt(mse)
# print("rmse for final is")
# print(rmse_final)

ntrain = np.shape(X_pca_train)[0]
ntest = np.shape(X_pca_test)[0]


kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, X_pca_train, Y):
        self.clf.fit(X_pca_train, Y)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, X_pca_train, Y):
        dtrain = xgb.DMatrix(X_pca_train, label=Y)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_pca_train[train_index]
        y_tr = Y[train_index]
        x_te = X_pca_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_pca_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}

gbm_params={
 'n_estimators': 400, 
 'max_depth': 10,
 'learning_rate': 0.35, 
 'loss': 'huber',
 'alpha':0.99
 }

xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)
gbm = SklearnWrapper(clf=GradientBoostingRegressor, seed=SEED, params=gbm_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)
gbm_oof_train, gbm_oof_test = get_oof(gbm)

print("XG-CV: {}".format(sqrt(mean_squared_error(Y, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(Y, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(Y, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(Y, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(Y, ls_oof_train))))
print("GBM-CV: {}".format(sqrt(mean_squared_error(Y, gbm_oof_train))))


X_pca_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train, gbm_oof_train), axis=1)
X_pca_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test, gbm_oof_test), axis=1)

print("{},{}".format(X_pca_train.shape, X_pca_test.shape))

dtrain = xgb.DMatrix(X_pca_train, label=Y)
dtest = xgb.DMatrix(X_pca_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
saleprice = np.exp(submission['SalePrice'])-1
submission['SalePrice'] = saleprice
submission.to_csv('xgstacker.csv', index=None)