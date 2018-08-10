#Creating a Stacking ensemble!
def model(x,y,z):
    import numpy as np
    from sklearn.cross_validation import KFold
    kf = KFold(x.shape[0], n_folds=5, random_state=0)
    n_train = x.shape[0]
    n_test  = z.shape[0]
    # Class to extend the Sklearn classifier
    class SklearnHelper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)
        
        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)
        
        def predict(self, x):
            return self.clf.predict(x)
        
        def fit(self,x,y):
            return self.clf.fit(x,y)
        
        def feature_importances(self,x,y):
            print(self.clf.fit(x,y).feature_importances_)
    
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
    import xgboost as xgb
    
    #Classifiers
    # Put in our parameters for said classifiers
    # Random Forest parameters
    rf_params = {
            'n_jobs': -1,
            'n_estimators': 200,
            'max_depth': 7,
            'max_features' : 'log2'
            }
    # Extra Trees Parameters
    et_params = {
            'n_jobs': -1,
            'n_estimators':200,
            #'max_features': 0.5,
            'max_depth': 8
            }
    # AdaBoost parameters
    ada_params = {
            'n_estimators': 25
            }
    # Gradient Boosting parameters
    gb_params = {
            'n_estimators': 600,
            'max_depth': 2
            }
    # Support Vector Classifier parameters 
    svc_params = {
            'kernel' : 'linear',
            'C' : 0.1
            }
    SEED = 0 #for reproducibility
    # Create 5 objects that represent our 4 models
    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((n_train,))
        oof_test = np.zeros((n_test,))
        
        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
            clf.fit(x_tr, y_tr)
            oof_train[test_index] = clf.predict(x_te)
        clf.fit(x_train,y_train)
        oof_test[:] = clf.predict(x_test)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    et_oof_train, et_oof_test = get_oof(et, x, y, z) # Extra Trees
    rf_oof_train, rf_oof_test = get_oof(rf,x, y, z) # Random Forest
    ada_oof_train, ada_oof_test = get_oof(ada, x, y, z) # AdaBoost 
    gb_oof_train, gb_oof_test = get_oof(gb,x, y, z) # Gradient Boost
    svc_oof_train, svc_oof_test = get_oof(svc,x, y, z) # Support Vector Classifier
    print("Training is complete")
    
    x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
    gbm = xgb.XGBClassifier(
            n_estimators= 2000,
            #learning_rate = 0.02,
            max_depth= 4,
            min_child_weight= 2,
            gamma=0.9,                        
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic',
            nthread= -1,
            scale_pos_weight=1).fit(x_train, y)
    predictions = gbm.predict(x_test)
    return predictions
    

