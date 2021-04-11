#Flask API

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import model as pickefiles

from pathlib import Path as path


model_path = path(r'model/sentiment_model.pkl')
recommend_path = path(r'data/Recommendations.pkl')
pred_path = path(r'data/pred_df.pkl')


# Create Flask object
app = Flask(__name__)  

# # Loading the dataset with pre-processed reviews
# with open('data/reviews_df.pkl', 'rb') as data:
#     reviews_df = pickle.load(data)
 
# if not model_path.exists() or not pred_path.exists():
#     pickefiles.prediction(reviews_df) 

# if not recommend_path.exists():
#     pickefiles.recommendation(reviews_df)  # Create the recommendation dataframe in a pickle file

# # Load the model from the file
# with open('model/sentiment_model.pkl', 'rb') as mod:
#     model = pickle.load(mod)

# # Load the sample data set with tf-idf values
# with open(r'data/pred_df.pkl', 'rb') as df:
#     pred_df = pickle.load(df)

# # Load the recommendation matrix 
# with open(r'data/Recommendations.pkl' , 'rb') as re:
#     recommendations =  pickle.load(re)

@app.route('/')
def home():
    return render_template('index.html', len = 0, message = '')



@app.route('/recommend', methods = ['POST'])
def recommend():
    # if(request.method =='POST'):

        # try:
        #     # Get values from browser
        #     user_id = request.form.get('user_id')

        #     # Recommending the Top 15 products to the user.
        #     top20 = recommendations.loc[user_id].sort_values(ascending = False).head(20)
        #     top20_df = pd.DataFrame(top20.reset_index(name = 'rating'))

        #     # get the dataset with tf-idf values for the Top 15 recommended products
        #     pred_df_temp = pred_df[pred_df.id.isin(top20_df.id.to_list())]

        #     # Predict the sentiments for these products using the sentiment model
        #     sentiment = model.predict(pred_df_temp[pred_df_temp.columns.difference(['id', 'name', 'reviews'])])
        #     sentiment_df = pd.DataFrame(sentiment, columns = ['sentiment'])

        #     pred_df_temp = pd.concat([pred_df_temp.reset_index(drop = True) , sentiment_df.reset_index(drop = True)], axis = 1)

        #     results_df = pred_df_temp.groupby(['id', 'name']).agg(sum = ('sentiment' , 'sum'), count = ('id', 'count')).reset_index().sort_values(by = 'sum' ,ascending = False)
        #     results_df['Positive Sentiment %'] = round((results_df['sum']/ results_df['count'] * 100 ),2)
        #     results_df = pd.merge(top20_df, results_df , on = 'id', how = 'left')

        #     result = results_df.sort_values(by = ['Positive Sentiment %'], ascending = False).head(5).to_dict(orient = 'records')

        #     return render_template('index.html', len = len(result), result = result, message = 'TOP 5 recommended products for User: ' + user_id)

        # except Exception as err:
        #     # message = err
        #     message = 'Invalid User! Try again'
        #     return render_template('index.html', len = 0, message = message)

    # else:
       return render_template('index.html', len = 0,message = '') 
    


if __name__ == '__main__' :
    # Start Application
    app.run()
    #  app.run(debug=True )  
    



