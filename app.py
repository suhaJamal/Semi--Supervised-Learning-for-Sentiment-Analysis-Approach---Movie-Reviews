import numpy as np
import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
import nltk
import requests
from textblob import TextBlob
import re
import gensim
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#nltk.download('stopwords')

if not nltk.corpus.stopwords.words('english'):
    nltk.download('stopwords')

API_KEY = '15b4c4f4c1c5a4b0deec395ada2a056c'

# Load the saved models
with open('models/logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)
with open('models/multinomial_nb_model.pkl', 'rb') as file:
    multinomial_model = pickle.load(file)
with open('models/xgboost_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)
with open('models/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# List of additional stopwords to consider
additional_stopwords = ["one", "like", "see", "make", "get", "really", "even", "time", "will", "much", "thing", "lot",
                        "going", "seem", "also", "still", "back", "want", "well", "br"]

stop_words = set(stopwords.words('english'))
TOP_RATED_MOVIES_URL = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page=1"
MIN_WORDS_THRESHOLD = 15

# Fetch top-rated movies
def fetch_top_rated_movies():
    response = requests.get(TOP_RATED_MOVIES_URL)
    if response.status_code == 200:
        data = response.json()
        return data['results']
    else:
        return []

top_rated_movies = fetch_top_rated_movies()
movie_titles = [movie['title'] for movie in top_rated_movies]

def fetch_movie_details(movie_id):
    """Fetch movie details by TMDb movie ID."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        #print("|||||||||| ", data)
        return {
            "title": data["title"],
            "poster_path": f"https://image.tmdb.org/t/p/w500{data['poster_path']}",
            "overview": data["overview"]
        }
    else:
        return None

def fetch_movie_reviews(movie_id):
    """Fetch movie reviews from TMDb."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return None

def preprocess_text(text):
    """Preprocess text by applying various cleaning and filtering operations."""
    # Lowercase conversion
    text = text.lower()
    # Remove HTML tags
    text = remove_html_tags(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Emoji removal
    text = remove_emoji(text)
    # Tokenize and remove stopwords using gensim's simple_preprocess and NLTK's stopwords
    words = simple_preprocess(text, deacc=True)  # deacc=True removes punctuations
    filtered_words = [word for word in words if word not in stop_words]
    # Join words back to a single string
    text = " ".join(filtered_words)
    return text

def remove_punc(text):
    exclude = set(string.punctuation)  # Assuming you've imported string.punctuation
    if isinstance(text, str):
        for char in exclude:
            text = text.replace(char, '')
    return text

def remove_emoji(text):
    # Check if the input is a string
    if not isinstance(text, (str, bytes)):
        # If not, return text as it is or convert it to a string as needed
        return text
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'' ,text)

def rem_tags(text):
    cleaned=re.sub("<.*?>","",text)
    return cleaned

def rem_stopwords(text):
    clean = []
    for word in text.split():
        if word.lower() not in stop_words:
            clean.append(word)
    return " ".join(clean)

def gensim_preprocess(text):
    cleaned = simple_preprocess(text)
    return cleaned

def vectorize_text(cleaned_text):
    return vectorizer.transform([cleaned_text])

def analyze_sentiment(review_text, model):
    """Analyze the sentiment of a review using the specified model."""
    cleaned_text = preprocess_text(review_text)
    review_vector = vectorize_text(cleaned_text)
    proba = None
    if hasattr(model, 'predict_proba'):
        prediction = model.predict(review_vector)[0]
        proba = model.predict_proba(review_vector)[0]
    else:
        raw_scores = model.predict(review_vector)
        prediction = int(raw_scores[0] >= 0.5)
        proba = np.array([1 - raw_scores[0], raw_scores[0]]) if raw_scores.ndim == 1 else raw_scores
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, proba


def display_wordcloud_positive(text):
    positive_words = []

    # Tokenize the input text and analyze each word's sentiment
    for word in text.split():
        word_sentiment = TextBlob(word).sentiment.polarity
        if word_sentiment > 0:  # Consider words with a positive sentiment score
            positive_words.append(word)

    # Join the positive words back into a string
    positive_text = " ".join(positive_words)

    # Generate a word cloud of positive words
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_font_size=110).generate(positive_text)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Use Streamlit to display the matplotlib plot
    st.pyplot(plt)

def display_wordcloud_negative(text):
    negative_words = []

    for word in text.split():
        word_sentiment = TextBlob(word).sentiment.polarity
        if word_sentiment < 0:  # Consider words with a negative sentiment score
            negative_words.append(word)

    negative_text = " ".join(negative_words)
    wordcloud = WordCloud(width=800, height=400, background_color='black', max_font_size=110, colormap='Reds').generate(
        negative_text)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def display_wordcloud(text):
    """Generate and display a word cloud from the given text."""
    stop_words = set(stopwords.words('english') + additional_stopwords)
    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words,
                          min_font_size=10).generate(text)
    plt.figure(figsize=(8, 8), dpi=80)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.show()
    st.pyplot(plt)

def main():
    st.title("Sentiment Analysis and Review Exploration")

    # Instruction on how to navigate through the app
    st.sidebar.header("Navigation")
    st.sidebar.write("Use the sidebar to navigate between different features of the app.")

    # Sidebar for activity selection
    activities = ["Review Analysis From TMDB", "Generate Word Cloud"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    # Model selection
    models = ["Logistic Regression", "Multinomial Naive Bayes", "XG Boost"]

    if choice == "Review Analysis From TMDB":
        st.header("Review Analysis From TMDB")
        st.write(
            "Analyze movie reviews and explore sentiments. Select a movie by its TMDb ID or choose from the top-rated movies.")

        model_choice = st.selectbox("Choose Model for Sentiment Analysis", models)

        # Based on the model choice, set the model to be used
        model = {
            "Logistic Regression": logistic_regression_model,
            "Multinomial Naive Bayes": multinomial_model,
            "XG Boost": xgboost_model
        }.get(model_choice)

        col_1, col_2 = st.columns([1, 2])
        with col_1:
            movie_id = st.text_input("Enter TMDb Movie ID:", key="movie_id")
            submit_id = st.button("Load ID")
        with col_2:
            # Initially load movie_titles and top_rated_movies somewhere above this code
            selected_movie_title = st.selectbox('Or select a movie to see reviews:',
                                                ['Select a movie...'] + movie_titles, key="selected_movie")
            # This will only fetch the ID if a movie is selected from the dropdown that isn't the placeholder
            if selected_movie_title != 'Select a movie...':
                selected_movie_id = next(
                    (movie['id'] for movie in top_rated_movies if movie['title'] == selected_movie_title), None)
            else:
                selected_movie_id = None

        # Logic to determine which movie ID to use
        if submit_id and movie_id:
            # User has inputted an ID and clicked "Load ID"
            st.session_state.final_movie_id = movie_id
            print("true.....")
        elif selected_movie_id:
            # User has selected a movie from the dropdown
            st.session_state.final_movie_id = selected_movie_id
        else:
            final_movie_id = None

        if 'final_movie_id' in st.session_state and st.session_state.final_movie_id:
            final_movie_id = st.session_state.final_movie_id
            movie_details = fetch_movie_details(final_movie_id)
            #print("=======  ", movie_details)
            reviews = fetch_movie_reviews(final_movie_id)
            st.session_state.reviews = reviews

            if movie_details and reviews:
                # Layout movie details and reviews side by side
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(movie_details["poster_path"])
                with col2:
                    st.subheader(movie_details["title"])
                    st.write(movie_details["overview"])
                    #st.subheader(movie_details["overview"])

                if 'review_index' not in st.session_state or st.session_state.review_index >= len(reviews):
                    st.session_state.review_index = 0

                review_index = st.session_state.review_index
                print("1- ", review_index)

                # Review navigation and display
                analyze, prev, next_button = st.columns([2,2,2])
                if prev.button("Previous Review"):
                    st.session_state.review_index = max(0, review_index - 1)
                    print("prev- ", review_index)
                if next_button.button("Next Review"):
                    st.session_state.review_index = min(len(reviews) - 1, review_index + 1)
                    print("Next- ", review_index)

                # Check if review_index is within the bounds and reviews are available
                if reviews and 0 <= review_index < len(reviews):  # Check if within bounds
                    selected_review = reviews[st.session_state.review_index]['content']
                    st.text_area("Review", value=selected_review, height=200)

                    # Place your "Analyze Sentiment" button and its functionality here
                    if analyze.button("Analyze Sentiment"):
                        sentiment, proba = analyze_sentiment(selected_review, model)
                        st.write(f"Sentiment: {sentiment}")
                        if proba is not None and len(proba) == 2:
                            st.write(f"Confidence - Positive: {proba[1]:.2f}, Negative: {proba[0]:.2f}")
                        st.session_state['last_review_text'] = selected_review
                        st.session_state['proba'] = proba
                        st.session_state['sentiment'] = sentiment

                else:
                    st.write("No reviews available for this movie.")

    elif choice == "Generate Word Cloud":
        st.header("Generate Word Cloud from Reviews")
        st.write("Create a word cloud visualization from a movie review. Depending on the sentiment analysis, a positive or negative word cloud will be generated.")
        # Use last review if available
        default_text = st.session_state.get('last_review_text', '')
        proba = st.session_state.get('proba', [0.5, 0.5])  # Default to neutral if not available
        sentiment = st.session_state.get('sentiment')
        raw_text = st.text_area("Review Text", value=default_text, height=250)

        if st.button("Generate Word Cloud Based on Sentiment"):
            if raw_text:
                if len(raw_text.split()) < MIN_WORDS_THRESHOLD:
                    st.write(
                        "This review is too short for a meaningful word cloud. More text is needed for richer insights.")
                else:
                    print("sent = ", sentiment)
                    cleaned_text = preprocess_text(raw_text)
                    #dominant_sentiment = get_dominant_sentiment(proba)
                    dominant_sentiment = sentiment
                    if dominant_sentiment == 'Positive':
                        st.subheader(f"{dominant_sentiment.capitalize()} Word Cloud for Selected Review")
                        display_wordcloud_positive(cleaned_text)
                    elif dominant_sentiment == 'Negative':
                        st.subheader(f"{dominant_sentiment.capitalize()} Word Cloud for Selected Review")
                        display_wordcloud_negative(cleaned_text)
                    else:
                        st.warning("The sentiment is too neutral for a meaningful word cloud. Please adjust the review text or analyze a different review.")
                        #display_wordcloud(cleaned_text)
            else:
                st.warning("Please enter text to generate word cloud.")

if __name__ == '__main__':
    main()
