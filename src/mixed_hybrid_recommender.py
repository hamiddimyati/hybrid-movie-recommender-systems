from recommender.content_based_recommender import train_content, predict_content
from recommender.matrix_fact_collab_recommender import train_mf, predict_mf

###################################################################################
#  Public API
###################################################################################
def train_mixed(data, catalog, params):
    model = MixedHybridRecommender()
    model.train(data, catalog, params)
    return model

def predict_mixed(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################
class MixedHybridRecommender():
    
    def train(self, data, catalog, params):
        self.data = data
        self.content_model  = train_content(data, catalog, params)
        self.mf_model = train_mf(data, catalog, params)
        self.limit = params['lmt']

    def predict(self, test):
        pred_content = predict_content(self.content_model, test, self.data)
        pred_mf = predict_mf(self.mf_model, test, self.data)
        
        merged = pred_mf.merge(pred_content, on=['userID', 'itemID'], how="outer").fillna(0)
        
        #set default prediction to pred_mf
        merged['prediction'] = merged['prediction_x']
        
        #set prediction to pred_content in top content dataframe
        top_content = merged.set_index(['itemID','prediction_x','prediction']).groupby('userID')['prediction_y'].nlargest(self.limit).reset_index() 
        top_content['prediction'] = top_content['prediction_y']
        
        #update highest pred_content to original pred_mf
        new_pred = merged[['userID','itemID','prediction']].set_index(['userID','itemID'])
        new_pred.update(top_content.set_index(['userID','itemID']))
        merged['prediction'] = new_pred.values
        
        return merged