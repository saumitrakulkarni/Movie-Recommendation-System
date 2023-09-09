#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
movies = pd.read_csv("data/movies.csv")


# In[2]:


movies


# In[3]:


import re

def clean_movie_titles(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# In[4]:


movies['cleaned_titles'] = movies['title'].apply(clean_movie_titles)


# # Building Term Frequency Matrix

# In[5]:


pip install sklearn


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf =  vectorizer.fit_transform(movies['cleaned_titles'])


# In[40]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    #title = 'Toy Story 1995'
    title = clean_movie_titles(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results


# In[53]:


import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
            value = 'Toy Story',
            description = 'Movie Title:',
            disabled=False
            )

movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data['new']
        if len(title)>=5:
            display(search(title))
        
        
movie_input.observe(on_type, names='value')

display(movie_input, movie_list)


# In[54]:


ratings = pd.read_csv("data/ratings.csv")


# In[58]:


ratings.dtypes


# In[59]:


similar_users = ratings[(ratings["movieId"]==1) & (ratings["rating"]>4)]["userId"].unique()


# In[60]:


similar_users


# In[67]:


similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"]>4)]["movieId"]


# In[70]:


similar_user_recs = similar_user_recs.value_counts()/len(similar_users)
similar_user_recs = similar_user_recs[similar_user_recs > .1]


# In[76]:


all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"]>4)]


# In[78]:


all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())


# In[79]:


all_users_recs


# In[82]:


rec_percent = pd.concat([similar_user_recs, all_users_recs], axis=1)
rec_percent.columns = ["similar", "all"]
rec_percent


# In[83]:


rec_percent["score"] = rec_percent["similar"]/rec_percent["all"]
rec_percent = rec_percent.sort_values("score", ascending=False)
rec_percent


# In[84]:


rec_percent.head(10).merge(movies, left_index=True, right_on="movieId")


# In[87]:


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"]==movie_id) & (ratings["rating"]>4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"]>4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts()/len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"]>4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percent = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percent.columns = ["similar", "all"]
    
    rec_percent["score"] = rec_percent["similar"]/rec_percent["all"]
    rec_percent = rec_percent.sort_values("score", ascending=False)
    
    return rec_percent.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title","genres"]]
    


# In[89]:


find_similar_movies(1)


# In[93]:


input_movie = widgets.Text(
    value="Toy Story",
    description="Movie Title:",
    disabled=False
)

recommedation_list = widgets.Output()

def on_type(data):
    with recommedation_list:
        recommedation_list.clear_output()
        title = data["new"]
        if len(title)>5:
            results = search(title)
            movieID = results.iloc[0]["movieId"]
            display(find_similar_movies(movieID))

input_movie.observe(on_type, names="value")
display(input_movie, recommedation_list)


# In[ ]:




