from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import scipy.sparse

###################################################################################
#  Public API
###################################################################################
def train_content(data, catalog, params):
    model = ContentBasedRecommender()
    model.train(data, catalog)
    return model

def predict_content(model, test, train):
    return model.predict(test)

###################################################################################
#  Helper Functions
###################################################################################
class ContentBasedRecommender():

    def train(self, data, catalog):
        self.data = data
        self.catalog = catalog

        #get feature vectors
        self.features = tf_idf_features(catalog)  #one_hot_features(catalog) #hybrid_features(catalog)
        
        #calculate similarities across all pairs 
        self.similarities = cosine_similarity(self.features)

    def predict(self, test):
        return get_relevance(self.data, test, self.catalog, self.similarities)


def cosine_similarity(matrix):
    # row represent one movie
    # normalize each row so that linear_kernel returns the cosine similarity
    l2norm = np.sqrt((matrix.multiply(matrix)).sum(axis=1))
    matrix_normalized = matrix / l2norm.reshape(matrix.shape[0],1)
    return linear_kernel(matrix_normalized, matrix_normalized) 

def tf_idf_features(catalog):
    catalog['words'] = catalog.genres.str.replace('|', ' ')\
        + ' ' + \
        catalog.title \
        + ' ' + \
        catalog.director1_names.fillna('') + ' ' + catalog.director2_names.fillna('') + ' ' + catalog.director3_names.fillna('')\
        + ' ' + \
        catalog.writer1_names.fillna('') +' ' +  catalog.writer2_names.fillna('')+ ' '+ catalog.writer3_names.fillna('')
    tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 1),min_df=0, stop_words='english')
    tf_idf_features = tfidf.fit_transform(catalog['words'])
    return tf_idf_features

def one_hot_features(catalog):
    enc_directors = OneHotEncoder(handle_unknown='ignore')
    enc_directors.fit(catalog[['director1_names']].fillna('not present').values)
    one_hot_directors1 = enc_directors.transform(catalog[['director1_names']].fillna('not present').values)
    one_hot_directors2 = enc_directors.transform(catalog[['director2_names']].fillna('not present').values)
    one_hot_directors3 = enc_directors.transform(catalog[['director3_names']].fillna('not present').values)
    enc_writers = OneHotEncoder(handle_unknown='ignore')
    enc_writers.fit(catalog[['writer1_names']].fillna('not present').values)
    one_hot_writers1 = enc_writers.transform(catalog[['writer1_names']].fillna('not present').values)
    one_hot_writers2 = enc_writers.transform(catalog[['writer2_names']].fillna('not present').values)
    one_hot_writers3 = enc_writers.transform(catalog[['writer3_names']].fillna('not present').values)
    one_hot_genres = catalog[['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
    one_hot_genres = scipy.sparse.csr_matrix(one_hot_genres.values)
    features = scipy.sparse.hstack([one_hot_genres, one_hot_directors1, one_hot_writers1])
    return features

def hybrid_features(catalog):
    oh_feature = one_hot_features(catalog)
    tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 1),min_df=0, stop_words='english')
    tfidf_features = tfidf.fit_transform(catalog.title)
    return scipy.sparse.hstack([tfidf_features, oh_feature])

def get_user_relevance(user_id, neighbors_of_u, train_ratings_of_u, test_ratings_of_u, similarities, catalog):
    
    # calculate the average user rating 
    avg_rating = np.average(train_ratings_of_u[train_ratings_of_u != 0])
    
    # for each new movie calculate a rating 
    # new items is a matrix with dim (New Items x length of N(U)) where each item is a similarity score
    neighbors_filter = catalog['movieId'].isin(neighbors_of_u.index.values)
    new_items = similarities[~neighbors_filter]
    new_items = new_items.T[neighbors_filter].T

    true_rating = train_ratings_of_u[neighbors_filter]
    normalizer = new_items.sum(axis=1) + 1e-5
    estimated_rating = avg_rating + new_items.dot((true_rating - avg_rating))/normalizer

    movies = catalog['movieId'][~neighbors_filter].values

    return pd.DataFrame(zip([user_id] * len(estimated_rating), movies, estimated_rating),
                        columns=['userID', 'itemID', 'prediction'])
    
def get_relevance(ratings_train, ratings_test, catalog, similarities):
    # get all movies rated by users
    neighbors_of_all_users = ratings_train.pivot(index='userID', columns='itemID', values='rating').fillna(0)
    movies_of_all_users = neighbors_of_all_users[neighbors_of_all_users != 0]
    # true rating of all movies in training data
    temp = ratings_train.pivot(index='itemID', columns='userID', values='rating').fillna(0)
    all_train_ratings = catalog[['movieId','title']]\
    .merge(temp, left_on='movieId', right_on='itemID', how='outer')\
    .drop(columns=['title','movieId'])\
    .fillna(0)\
    .transpose()
    
    # true rating of all movies in testing data
    temp = ratings_test.pivot(index='itemID', columns='userID', values='rating').fillna(0)
    all_test_ratings = catalog[['movieId','title']]\
    .merge(temp, left_on='movieId', right_on='itemID', how='outer')\
    .drop(columns=['title','movieId'])\
    .fillna(0)\
    .transpose()
    
    rating_pred = pd.DataFrame(columns=['userID', 'itemID', 'prediction'])
    
    for user_id in list(set(ratings_train.userID) & set(ratings_test.userID)):
        neighbors_of_u = neighbors_of_all_users.loc[user_id]
        neighbors_of_u = neighbors_of_u[neighbors_of_u != 0]
        train_ratings_of_u = all_train_ratings.loc[user_id].values
        test_ratings_of_u = all_test_ratings.loc[user_id].values
        df = get_user_relevance(user_id, neighbors_of_u, train_ratings_of_u, test_ratings_of_u, similarities, catalog)
        rating_pred = pd.concat([rating_pred, df])
        print(user_id, end =', ')
    return rating_pred.astype({'userID': 'int32', 'itemID': 'int32'})