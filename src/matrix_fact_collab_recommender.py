import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
###################################################################################
#  Public API
###################################################################################
def train_mf(data, catalog, params):
    model = MatrixFactorizationCollabRecommender()
    model.train(data, catalog, params)
    return model 

def predict_mf(model, test, train, remove_train=True):
    return model.predict(test, remove_train=remove_train)

###################################################################################
#  Helper Functions
###################################################################################
class MatrixFactorizationCollabRecommender():

    def train(self, data, catalog, params):
        self.data = data
        self.ratings_df = data.pivot(index = 'userID', columns ='itemID', values = 'rating').fillna(0)

        alpha = params['alpha']
        l1_ratio = params['l1_ratio']
        self.model = NMF(n_components=50, init='random', alpha=alpha, l1_ratio=l1_ratio)
        self.W = self.model.fit_transform(self.ratings_df.values)
        self.H = self.model.components_

    def predict(self, test, remove_train):
        
        all_user_predicted_ratings = np.matmul(self.W, self.H)

        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = self.ratings_df.columns)
        preds_df.reset_index(inplace=True)
        preds_df = pd.melt(preds_df, id_vars=['index'], var_name=['itemID'])
        preds_df.columns = ['userID', 'itemID', 'prediction']
        preds_df['userID'] = preds_df['userID'] + 1 #userID starts from 1
        merged = preds_df
        
        if remove_train:
            # remove training data from prediction
            tempdf = pd.concat(
                [
                    self.data[['userID', 'itemID']],
                    pd.DataFrame(
                        data=np.ones(self.data.shape[0]), columns=["dummycol"], index=self.data.index
                    ),
                ],
                axis=1,
            )
            merged = pd.merge(tempdf, preds_df, on=['userID', 'itemID'], how="outer")
            merged = merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)

        return merged.astype({'itemID': 'int32'}) 