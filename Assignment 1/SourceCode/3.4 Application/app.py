import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import LinearSVC

df_json = pd.read_json('reviewSelected100.json', lines=True)
restaurant_df = pd.DataFrame(df_json)
reviews = []
for i in range(len(restaurant_df)):
    reviews.append(
        {'text': restaurant_df['text'][i], 'stars': int(restaurant_df['stars'][i])})
reviews_df = pd.DataFrame(reviews)
reviews_df = reviews_df.sample(frac=1)
reviews_df = reviews_df.reset_index(drop=True)
print("\n*********Data Prepared*********\n\n")
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Helper function to categorize sentiment to negative, neutral, positive based on star rating


def classify_rating(value):
    value = int(value)
    if value >= 4:
        return 2
    if value == 3:
        return 1
    else:
        return 0


reviews_df['Sentiment_Classification'] = reviews_df['stars'].apply(
    classify_rating)

# Defining different sentiment category
sentiment_label = ['negative', 'neutral', 'positive']

# creates the sentiment classifier


def analyse_string_input(data):
    cv = CountVectorizer(stop_words="english", ngram_range=(
        1, 3), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(reviews_df.text.tolist()+[data])
    train_reviews_counts = text_counts[0:12240]
    test_reviews_counts = text_counts[12240:15297]
    input_data_counts = text_counts[-1]
    train_stars = reviews_df['Sentiment_Classification'][0:12240]
    test_stars = reviews_df['Sentiment_Classification'][12240:]
    x_train = train_reviews_counts
    y_train = train_stars
    LSVC = LinearSVC()
    LSVC.fit(x_train, y_train)
    return LSVC.predict(input_data_counts)

# Takes user input and gives output


def perform_analysis():
    sentence = str(input("\n\nPlease enter the text you want to analyse: "))
    print("-"*100)
    print("The input entered by you is:", sentence)
    print('-'*100)
    print("Analyzing your input.....")
    print("-" * 100)
    print("Analaysis Result: Your input is classified to have a",
          sentiment_label[analyse_string_input(sentence)[0]], "sentiment.")
    print('-'*100)


def main():
    print("Welcome to our Sentiment Classifier!")
    print("Press 1 to analyse a sentence")
    print("Press 2 to exit")
    choice = str(input("Enter your choice:"))
    if choice == '1':
        perform_analysis()
    while True:
        print("To try another sentence, enter 1 again.")
        print("Press any other character to exit")
        new_choice = str(input("Enter your choice:"))
        if new_choice == '1':
            perform_analysis()
        else:
            print("Thank you for using our application!")
            break


if __name__ == "__main__":
    main()
