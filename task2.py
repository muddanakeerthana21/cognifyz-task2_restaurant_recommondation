import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv("Dataset .csv")

# Handle Missing Values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Feature Selection
features = df["Cuisines"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(features)

# Similarity Matrix
similarity = cosine_similarity(tfidf_matrix)

# Recommendation Function
def recommend_restaurant(name):

    restaurant_index = df[
        df["Restaurant Name"] == name
    ].index[0]

    similarity_scores = list(
        enumerate(similarity[restaurant_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended Restaurants:\n")

    for i in similarity_scores[1:6]:
        print(
            df.iloc[i[0]]["Restaurant Name"]
        )

# Example
recommend_restaurant("Ooma")