import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

###################################################################################
#  Public API
###################################################################################
def train_item(data, catalog):
    model = ItemItemCollabRecommender()
    model.train(data, catalog)
    return model 

def predict_item(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################
class ItemItemCollabRecommender():

    def train(self, data, catalog):
        self.data = data
        self.catalog = catalog
        self.ratings_matrix_items = get_ratings_matrix(data)
        
    def predict(self, test):
        return recommend_all_users(self.data, test, self.catalog, self.ratings_matrix_items)

def recommend_all_users(train, test, catalog, ratings_matrix_items):
    rating_pred = pd.DataFrame(columns=['userID', 'itemID', 'prediction'])
    for user_id in np.unique(test['userID']):
        df = recommend_movies_as_per_item_similarity(user_id, catalog, train, ratings_matrix_items)
        rating_pred = pd.concat([rating_pred, df])
        print(user_id, end =', ')

    return rating_pred.astype({'userID': 'int32', 'itemID': 'int32'})    

def get_ratings_matrix(df_movies_ratings):
    ratings_matrix_items = df_movies_ratings.pivot_table(index=['itemID'],columns=['userID'],values='rating').reset_index(drop=True)
    ratings_matrix_items.fillna( 0, inplace = True )
    movie_similarity = 1 - pairwise_distances( ratings_matrix_items.iloc[:,:].values, metric="cosine" )
    np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
    ratings_matrix_items = pd.DataFrame(movie_similarity)
    return ratings_matrix_items

def recommend_movies_as_per_item_similarity(user_id, catalog, df_movies_ratings, ratings_matrix_items):
    """
    Recommending movie which user hasn't watched as per Item Similarity
    :input user_id: user_id to whom movie needs to be recommended
    """
    user_movies = df_movies_ratings[(df_movies_ratings['userID']==user_id)]
    watched = catalog['movieId'].isin(user_movies['itemID'])
    new_movies = catalog['movieId'][~watched]
    similarities = ratings_matrix_items.iloc[catalog['movieId'][watched].index]
    similarities = similarities.T.iloc[new_movies.index].T

    normalizer = (similarities.values.sum(axis=0) + 1e-5)
    estimated_rating = (user_movies.rating.values.reshape(-1,1) * similarities.values).sum(axis=0)/normalizer
    return pd.DataFrame(zip([user_id] * len(estimated_rating), new_movies.values, estimated_rating),
                        columns=['userID', 'itemID', 'prediction'])