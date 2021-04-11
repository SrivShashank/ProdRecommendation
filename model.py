import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

import pickle

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


def recommendation(reviews_df):
    # Using Item-Item Recommendation system
    # we need only id , reviews_rating, reviews_username columns for our analysis
    user_rating = reviews_df[['id', 'reviews_rating', 'reviews_username']]

    # converting user id to lower cases
    user_rating['reviews_username'] = user_rating['reviews_username'].apply(lambda x: str(x).lower())

    # dropping the duplicates
    user_rating.drop_duplicates(['id', 'reviews_username'], inplace=True)

    # Pivot the dataset into matrix format where columns are products and the rows are user IDs.
    df_pivot_final = user_rating.pivot(
        index='id',
        columns='reviews_username',
        values='reviews_rating'
    )

    # Creating the Item-item Similarity Matrix using cosine similarity function.
    correlation_final = cosine_similarity(df_pivot_final.fillna(0))
    final_ratings_predicted = np.dot(
        df_pivot_final.fillna(0).T, correlation_final)

    dummy_final = user_rating.copy()
    # The product rated by the user is marked as 0 and product NOT rated by user is marked as 1 for recommendation.
    dummy_final['reviews_rating'] = dummy_final['reviews_rating'].apply(
        lambda x: 0 if x >= 1 else 1)
    dummy_final = dummy_final.pivot(
        index='reviews_username', columns='id', values='reviews_rating').fillna(1)

    # Filtering the rating only for the products NOT rated by the user for recommendation
    final_rating = np.multiply(final_ratings_predicted, dummy_final)

    # save final_rating to a pickle
    with open('data/Recommendations.pkl', 'wb') as rec:
        pickle.dump(final_rating, rec, protocol=pickle.HIGHEST_PROTOCOL)


def prediction(reviews_df):
    #  1. Build Logistic Regression model using tf-idf with top 5000 features
    #  2. Create a new dataframe with all the reviews and their tf-idf representation. This wil be used for prediction

    # Creating Train and Test datasets
    X = reviews_df['reviews']
    y = reviews_df['user_sentiment']

    # splitting the data into Train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=100)

    # Using TF-IDF for Feature extraction
    tfidf_vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf_train = tfidf_vec.fit_transform(X_train)
    X_tfidf_test = tfidf_vec.transform(X_test)

    # Using SMOTE to balance the train data
    oversample = SMOTE()
    X_tfidf_train, y_tfidf_train = oversample.fit_resample(
        X_tfidf_train, y_train)

    # Using Logistic Regression on TFIDF
    logreg = LogisticRegression(C=5, random_state=42,solver='liblinear')
    logreg.fit(X_tfidf_train, y_tfidf_train)

    # Saving the model in a pickle file
    with open('model/sentiment_model.pkl', 'wb') as mod:
        pickle.dump(logreg, mod, protocol=pickle.HIGHEST_PROTOCOL)

    # creating a Dataframe with the Product Id , name  and  pre-processed reviews text
    prediction_df = pd.concat([reviews_df[['id', 'name']].reset_index(
        drop=True), X.reset_index(drop=True)], axis=1)

    # creating a data frame with tf-idf values for all the reviews
    reviews_tfidf = tfidf_vec.transform(prediction_df['reviews'])

    # adding the tf-idf values to the prediction_df
    prediction_df = pd.concat([prediction_df.reset_index(drop=True), pd.DataFrame(
        reviews_tfidf.toarray()).reset_index(drop=True)], axis=1)

    # saving the prediction_df in a pickle file
    with open('data/pred_df.pkl', 'wb') as df:
        pickle.dump(prediction_df, df, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    try:
        # Loading the dataset with pre-processed reviews
        with open('data/reviews_df.pkl', 'rb') as data:
            reviews_df = pickle.load(data)

        recommendation(reviews_df)  # Create the recommendation dataframe in a pickle file
        prediction(reviews_df)      # Build LR model and Create a new dataframe with all the reviews and their tf-idf representation
        return True
    except Exception as err:
        return err


if __name__ == '__main__':
    x = main()
    if x:      
        # Loading the dataset with pre-processed reviews
        with open('data/reviews_df.pkl', 'rb') as data:
            reviews_df = pickle.load(data)

        # Loading the dataset with recommendations
        with open('data/Recommendations.pkl', 'rb') as rec:
            recommend = pickle.load(rec)

        # Loading the dataset with tf-idf vectorized reviews
        with open('data/pred_df.pkl', 'rb') as pred:
            prediction_df = pickle.load(pred)

        # Loading the model
        with open('model/sentiment_model.pkl', 'rb') as mod:
            model = pickle.load(mod)

            while(True):
                # Take the user ID as input.
                user_input = str.lower(input("Enter your user name: "))

                # Validate user
                if (user_input not in (reviews_df.reviews_username.to_list())):
                    print('Invalid User!')

                else:
                    # Valid user
                    print('Getting the recommended products.... ')
                    # get top 20 recommendations
                    top20 = recommend.loc[user_input].sort_values(ascending=False)[
                    0:20]
                    top20_df = pd.DataFrame(top20.reset_index(name='rating'))

                    # predict the sentiments of the top 20 products
                    pred_df = prediction_df.copy()
                    pred_df = pred_df[pred_df.id.isin(top20_df.id.to_list())]
                    res = model.predict(
                    pred_df[pred_df.columns.difference(['id', 'name', 'reviews'])])
                    res_df = pd.DataFrame(res, columns=['sentiment'])
                    pred_df = pd.concat([pred_df.reset_index(
                        drop=True), res_df.reset_index(drop=True)], axis=1)
                    results_df = pred_df.groupby(['id', 'name']).agg(sum=('sentiment', 'sum'), count=(
                        'id', 'count')).reset_index().sort_values(by='sum', ascending=False)
                    results_df['Positive Sentiment %'] = round(
                         (results_df['sum'] / results_df['count'] * 100), 2)
                    results_df = pd.merge(
                    top20_df, results_df, on='id', how='left')

                    # Get the Top 5 out of the top 20 based on the 'Positive Sentiment %'
                    output = results_df.sort_values(
                        by=['Positive Sentiment %'], ascending=False).head()
                    print('Top 5 recommended products: ')
                    print(*[f'{index + 1}  {prod} ' for index,
                          prod in enumerate(output['name'].tolist())], sep='\n')

                    choice = str.lower(input('\n Do you want to continue (Y/N): '))
                    if choice != 'y':
                        break
    else:
        print('Error encountered:', x)
