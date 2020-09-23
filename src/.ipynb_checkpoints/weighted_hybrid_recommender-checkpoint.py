from recommender.content_based_recommender import train_content, predict_content
from recommender.matrix_fact_collab_recommender import train_mf, predict_mf
###################################################################################
#  Public API
###################################################################################
def train_weighted(data, catalog):
    model = WeightedHybridRecommender()
    model.train(data, catalog)
    return model

def predict_weighted(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################
class WeightedHybridRecommender():

    def __init__(self):
        self.alpha = 0.6

    def train(self, data, catalog):
        self.data = data
        self.content_model  = train_content(data, catalog)
        self.mf_model = train_mf(data, catalog)

    def predict(self, test):
        pred_content = predict_content(self.content_model, test, self.data)
        pred_mf = predict_mf(self.mf_model, test, self.data)

        merged = pred_mf.merge(pred_content, on=['userID', 'itemID'], how="outer").fillna(0)
        merged['prediction'] = self.alpha * merged['prediction_x'] + (1-self.alpha) * merged['prediction_y']
        return merged
