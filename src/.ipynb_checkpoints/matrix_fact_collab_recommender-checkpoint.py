import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
###################################################################################
#  Public API
###################################################################################
def train_mf(data, catalog):
    model = MatrixFactorizationCollabRecommender()
    model.train(data, catalog)
    return model 

def predict_mf(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################
class MatrixFactorizationCollabRecommender():

    def train(self, data, catalog):
        self.data = data
        self.ratings_df = data.pivot(index = 'userID', columns ='itemID', values = 'rating').fillna(0)
        self.model = NMF(n_components=50, init='random', random_state=0)
        self.W = self.model.fit_transform(self.ratings_df.values)
        self.H = self.model.components_

    def predict(self, test):
        
        all_user_predicted_ratings = np.matmul(self.W, self.H)

        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = self.ratings_df.columns)
        preds_df.reset_index(inplace=True)
        preds_df = pd.melt(preds_df, id_vars=['index'], var_name=['itemID'])
        preds_df.columns = ['userID', 'itemID', 'prediction']
        preds_df['userID'] = preds_df['userID'] + 1 #userID starts from 1
        """
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
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1).astype({'itemID': 'int32'}) 
        """
        return preds_df.astype({'itemID': 'int32'})