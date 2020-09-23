import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
###################################################################################
#  Public API
###################################################################################
def train_user(data, catalog, params):
    model = UserUserCollabRecommender()
    model.train(data, catalog)
    return model

def predict_user(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################

class UserUserCollabRecommender(): 
    def train(self, data, catalog): 
        self.data = data 
        self.catalog = catalog 
        self.ratings_matrix_users = get_ratings_matrix(data) 
        
    def predict(self, test):
        return recommend_all_users(self.data, test, self.catalog, self.ratings_matrix_users) 

def recommend_all_users(train, test, catalog, ratings_matrix_users):
    rating_pred = pd.DataFrame(columns=['userID', 'movieID', 'prediction'])
    for user_id in np.unique(test['userID']):
        df = recommend_movies_as_per_user_similarity(train, user_id, catalog, ratings_matrix_users)
        rating_pred = pd.concat([rating_pred, df])
        print(user_id, end=',')
    return rating_pred.astype({'userID': 'int32', 'movieID': 'int32'})

def get_ratings_matrix(df_movies_ratings):
    ratings_matrix_users = df_movies_ratings.pivot_table(index=['userID'], columns=['movieId'], values='rating').reset_index(drop=True)
    ratings_matrix_users.fillna(0, inplace=True)
    movie_similarity = 1 - pairwise_distances(ratings_matrix_users.iloc[:, :].values, metric="cosine")
    np.fill_diagonal(movie_similarity, 0)  # Filling diagonals with 0s for future use when sorting is done
    ratings_matrix_users = pd.DataFrame(movie_similarity)
    return ratings_matrix_users

def similar_user_series(ratings_matrix_users):
    df_similar_user = ratings_matrix_users.to_frame()
    df_similar_user.columns = ['similarUser']
    return df_similar_user


def recommend_movies_as_per_user_similarity(df_movies_ratings, user_id, catalog, df_similar_user):
    """
    Recommending movie which user hasn't watched as per User Similarity
    :input user_id: user_id to whom movie needs to be recommended
    """
    user_movies = df_movies_ratings[df_movies_ratings['userId'] == user_id]
    sim_user = df_similar_user.iloc[0, 0]
    movieId = df_movies_ratings['movieId'][0]
    watched = catalog['userId'].isin(sim_user)
    new_movies = df_movies_ratings[(df_movies_ratings.userId == sim_user) & (df_movies_ratings.movieId == movieId)]
    similarities = df_similar_user.iloc[catalog['userId'][watched].index]
    similarities = similarities.T.iloc[new_movies.index].T
    normalizer = (similarities.values.sum(axis=0) + 1e-5)
    estimated_rating = (user_movies.rating.values.reshape(-1, 1) * similarities.values).sum(axis=0) / normalizer
    return pd.DataFrame(zip([user_id] * len(estimated_rating), new_movies['movieId'], estimated_rating), columns=['user_id', 'movieId', 'prediction'])
