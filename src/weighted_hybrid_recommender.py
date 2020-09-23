from recommender.content_based_recommender import train_content, predict_content
from recommender.matrix_fact_collab_recommender import train_mf, predict_mf
###################################################################################
#  Public API
###################################################################################
def train_weighted(data, catalog, params):
    model = WeightedHybridRecommender()
    model.train(data, catalog, params)
    return model

def predict_weighted(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################
class WeightedHybridRecommender():

    def train(self, data, catalog, params):
        self.data = data
        self.content_model  = train_content(data, catalog, params)
        self.mf_model = train_mf(data, catalog, params)
        self.alpha = params['wght']

    def predict(self, test):
        pred_content = predict_content(self.content_model, test, self.data)
        pred_mf = predict_mf(self.mf_model, test, self.data)

        merged = pred_mf.merge(pred_content, on=['userID', 'itemID'], how="outer").fillna(0)
        merged['prediction'] = self.alpha * merged['prediction_x'] + (1-self.alpha) * merged['prediction_y']
        return merged
