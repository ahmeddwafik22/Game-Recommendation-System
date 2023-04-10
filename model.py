# Import modules
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

games_df = pd.read_csv('Games_dataset.csv', index_col=0)

print('Number of games loaded: %s ' % (len(games_df)), '\n')

# Display the data
games_df.head()


import nltk
nltk.download('punkt')
import re
from nltk.stem.snowball import SnowballStemmer
#nltk.download('punkt')

# Create an English language SnowballStemmer object
stemmer = SnowballStemmer("english")

# Define a function to perform both stemming and tokenization
def tokenize_and_stem(text):

    tokens = [word for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]


    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    stems = [stemmer.stem(word) for word in filtered_tokens]

    return stems


from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer object with stopwords and tokenizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))

# Fit and transform the tfidf_vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in games_df["Plots"]])
# ==================================KMeans==================================================================
from sklearn.cluster import KMeans

km = KMeans(n_clusters=7)

# Fit the k-means object with tfidf_matrix
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

games_df["cluster"] = clusters

similarity_distance = 1 - cosine_similarity(tfidf_matrix)
vals = games_df.Title.tolist()
similarity_df = pd.DataFrame(similarity_distance, columns=vals, index=vals)
# Export
similarity_df.to_csv('simlarity_matrix.csv')
