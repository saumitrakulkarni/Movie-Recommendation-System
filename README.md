# Movie Recommender

## Overview

Movie Recommender is a movie recommendation project that provides personalized movie suggestions to users based on a custom recommendation score. Unlike traditional movie recommendation systems that rely solely on popularity or user ratings, this system takes into account movies liked by other users with similar interests. The project uses the MovieLens 25M dataset as a source of movie information and recommendations.

## Methodology

The movie recommendation system operates as follows:

1. **Data Source**: The project utilizes the [MovieLens 25M dataset](https://files.grouplens.org/datasets/movielens/ml-25m.zip) as the primary data source. This dataset contains a vast amount of movie-related information, including user ratings and movie metadata.

2. **Data Cleaning**: The dataset is cleaned and preprocessed using the Pandas library. This step involves handling missing data, formatting, and other data preparation tasks. The `re` library is also employed for regular expression-based data cleaning when necessary.

3. **Recommendation Score Calculation**: The core of the project lies in the custom recommendation score calculation. When a user provides a movie of interest, the system calculates a recommendation score for other movies. This score is based not only on the movie's popularity but also on the preferences of users who have similar interests in movies. The exact details of the recommendation score calculation can be found in the code.

4. **User Interaction**: Users can interact with the system by entering a movie they like or are interested in. The system then calculates and displays a list of recommended movies based on the custom recommendation score.

## Dependencies

- Python
- Pandas library
- Regular Expressions (`re`)
- Scikit-learn (`sklearn`)
  - `TfidfVectorizer`: Used for transforming movie descriptions into TF-IDF vectors for text-based analysis.
  - `cosine_similarity`: Used to calculate the cosine similarity between TF-IDF vectors to measure similarity between movies.
- NumPy (`numpy`): Used for various numerical operations and data manipulation.
- Jupyter Widgets (`ipywidgets`): Used for interactive input in Jupyter Notebook.
- IPython.display (`IPython.display`): Used to display output within Jupyter Notebook.
