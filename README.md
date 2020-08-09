# Text Classification And Sentiment Analysis On Hotel Reviews

## Goal
The focus of this study is training supervised learning text classification models to see whether or not its possible to predict reviews ratings. The dataset was scraped from TripAdvisor and contained the name of a person leaving a review, the actual user review and the rating from the top ten rated hotels. After tokenizing, lemmatize and filtering the data, I vectorized the reviews first with the term frequency-inverse document frequency (tf-idf) method, which provides insight to the weight of each word in each document and also with count vectorizer that transforms the text in vectors of the tokens counts. 

## Data Collection
I scraped the data from the TripAdvisor website using Beautifull Soup and Selenium libraries. The dataset contains the reviews, users and ratings from hotels in TripAdvisor located in New York. The data set has 25,050 entries.


## Exploratory Data Anlaysis (EDA) 

Most of the reviews have a 5 stars that is 47% of the data. Only 7% of reviews have 1 star. 

There are 22,205 unique reviews in the data which is quite diverse. The average user has contributed with 1 review to the dataframe. The top reviewer (Marck C) has contributed 13 reviews.

The maximum number of characters in a review is 2,535 and the minimum number is 195 characters. The average length of characters in a review is 469 characters.

![](Images/review_frequency)
