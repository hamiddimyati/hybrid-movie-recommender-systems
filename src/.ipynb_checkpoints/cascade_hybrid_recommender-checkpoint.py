from recommender.content_based_recommender import train_content, predict_content
from recommender.matrix_fact_collab_recommender import train_mf, predict_mf
import numpy as np

###################################################################################
#  Public API
###################################################################################
def train_cascade(data, catalog):
    model = CascadeHybridRecommender()
    model.train(data, catalog)
    return model

def predict_cascade(model, test, train):
    return model.predict(test)


###################################################################################
#  Helper Functions
###################################################################################
class CascadeHybridRecommender():

    def __init__(self):
        self.threshold = 0.5
        
    def train(self, data, catalog):
        self.data = data
        self.content_model = train_content(data, catalog)
        pred_content = predict_content(self.content_model, self.data, self.data)
        pred_content['prediction'] = np.where(pred_content['prediction'] < self.threshold, 0, pred_content['prediction'])
        pred_content = pred_content.rename(columns={'prediction':'rating'})
        self.mf_model = train_mf(pred_content, catalog)
        
    def predict(self, test):
        pred_hybrid = predict_mf(self.mf_model, test, self.data)
        return pred_hybrid