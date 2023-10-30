
import numpy as np
import sklearn


class feature_methods():
    
    def __init__(self):
        """_summary_
        """
        
        
        
    def getNfeaturesANOVAF(self, n, X_train, y_train):
        res = []
        for colName in X_train.columns:
            if colName != "intercept":
                res.append(
                    (
                        colName,
                        sklearn.feature_selection.f_classif(
                            np.array(X_train[colName]).reshape(-1, 1), y_train
                        )[0],
                    )
                )
        sortedList = sorted(res, key=lambda X:X[1])
        sortedList.reverse()
        nFeatures = sortedList[:n]
        finalColNames = []
        for elem in nFeatures:
            finalColNames.append(elem[0])
        return finalColNames
    
    
    