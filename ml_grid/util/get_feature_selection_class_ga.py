




import random
from operator import itemgetter

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import f_classif


class feature_selection_methods_class():
    
    
    def __init__(self, ml_grid_object):
        
      
      self.X_train = ml_grid_object.X_train
      
      self.y_train = ml_grid_object.y_train
      
      self.X_test = ml_grid_object.X_test
      
      
      start_val = 2
      
      self.feature_parameter_vector = np.linspace(
            2, len(self.X_train.columns), int(len(self.X_train.columns) + 1 - start_val)
        ).astype(int)
      
      
      
    

    def getNfeaturesANOVAF(self, n):
        res = []
        for colName in self.X_train.columns:
            if colName != "intercept":
                # Compute F-score
                f_score = f_classif(
                    np.array(self.X_train[colName]).reshape(-1, 1), self.y_train
                )[0][0]

                # Skip if F-score is NaN
                if not np.isnan(f_score):
                    res.append((colName, f_score))

        # Sort by F-score descending
        sortedList = sorted(res, key=lambda x: x[1], reverse=True)

        # Take top n
        nFeatures = sortedList[:n]

        # Extract column names
        finalColNames = [elem[0] for elem in nFeatures]

        return finalColNames



    def getRandomForestFeatureColumns(self, X, y, n):
        try:
            X = X.drop("index", axis=1)
        except:
            pass
        independentVariables = X
        self.X_train = X
        self.y_train = y
        forest = RandomForestClassifier(random_state=1)
        forest.fit(self.X_train, self.y_train)
        importances = forest.feature_importances_
        namedFeatures = []
        for i in range(0, len(self.X_train.columns)):
            namedFeatures.append((list(self.X_train.columns)[i], importances[i]))
        sortedNamedFeatures = sorted(namedFeatures, key=itemgetter(1))
        sortedNamedFeatures.reverse()
        nFeatures = sortedNamedFeatures[:n]
        finalColNames = []
        for elem in nFeatures:
            finalColNames.append(elem[0])
        return finalColNames


    def getXGBoostFeatureColumns(self, X, y, n):
        try:
            X = X.drop("index", axis=1)
        except:
            pass
        try:
            model = XGBClassifier()
            model.fit(X, y)
            importances = model.feature_importances_
            namedFeatures = []
            for i in range(0, len(self.X_train.columns)):
                namedFeatures.append((list(self.X_train.columns)[i], importances[i]))
            sortedNamedFeatures = sorted(namedFeatures, key=itemgetter(1))
            sortedNamedFeatures.reverse()
            nFeatures = sortedNamedFeatures[:n]
            finalColNames = []
            for elem in nFeatures:
                finalColNames.append(elem[0])
        except Exception as e:
            print(e)
            print("Failed to get xgboost feature columns")
            raise e

        return finalColNames


    def getExtraTreesFeatureColumns(self, X, y, n):
        independentVariables = X

        self.X_train = X
        self.y_train = y
        try:
            X = X.drop("index", axis=1)
        except:
            X = X
        forest = ExtraTreesClassifier(random_state=1)
        forest.fit(self.X_train, self.y_train)
        importances = forest.feature_importances_
        namedFeatures = []
        for i in range(0, len(self.X_train.columns)):
            namedFeatures.append((list(self.X_train.columns)[i], importances[i]))
        sortedNamedFeatures = sorted(namedFeatures, key=itemgetter(1))
        sortedNamedFeatures.reverse()

        #     xFeatureColNames = []
        #     for i in range(0, n):
        #         xFeatureColNames.append(sortedNamedFeatures[i][0])
        nFeatures = sortedNamedFeatures[:n]
        finalColNames = []
        for elem in nFeatures:
            finalColNames.append(elem[0])
        return finalColNames


    # # Base learner and ensemble generation  
            




    

    def get_featured_selected_training_data(self, method='anova'):
        #global self.X_train, X_test, self.y_train, y_test
            # Select n features------------------------------------------------------------------------
        f = self.feature_parameter_vector 
        
        nFeatures = random.choice(f)
        
        if(method=='anova'):
            xFeatureColumnNames = feature_selection_methods_class.getNfeaturesANOVAF(self, 
                nFeatures)
        
        elif(method=='randomforest'):
            xFeatureColumnNames = feature_selection_methods_class.getRandomForestFeatureColumns(self, 
            self.X_train, self.y_train, nFeatures
        )
        elif(method=='extratrees'):
            xFeatureColumnNames = feature_selection_methods_class.getExtraTreesFeatureColumns(self,
            self.X_train, self.y_train, nFeatures
        )
        elif(method=='xgb'):
            xFeatureColumnNames = feature_selection_methods_class.getXGBoostFeatureColumns(self,
            self.X_train, self.y_train, nFeatures
        )
            
            
        X_train_fs = self.X_train[xFeatureColumnNames].copy()
        X_test_fs = self.X_test[xFeatureColumnNames].copy()
        
        return X_train_fs, X_test_fs

