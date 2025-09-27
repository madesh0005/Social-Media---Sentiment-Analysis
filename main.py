# Social Media Sentiment Analysis Project

# Install dependencies inside Jupyter (if not already installed)
import sys
!{sys.executable} -m pip install wordcloud textblob nltk matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk

# Download stopwords for text cleaning
nltk.download("stopwords")

from nltk.corpus import stopwords

# ----------------------------
# Step 1: Take user input
# ----------------------------
user_text = input("Enter a social media post/comment: ")

# ----------------------------
# Step 2: Sentiment Analysis
# ----------------------------
analysis = TextBlob(user_text)
sentiment_score = analysis.sentiment.polarity  # -1 to +1

if sentiment_score > 0:
    sentiment = "Positive 😀"
    sentiment_color = "green"
elif sentiment_score < 0:
    sentiment = "Negative 😡"
    sentiment_color = "red"
else:
    sentiment = "Neutral 😐"
    sentiment_color = "blue"

print("\n📝 Your Input: ", user_text)
print("📊 Sentiment Score: ", sentiment_score)
print("✅ Classified as: ", sentiment)

# ----------------------------
# Step 3: Visualization
# ----------------------------

# 1. Sentiment Bar Chart
plt.figure(figsize=(6,4))

# If sentiment is neutral, we create a very small positive bar to display the neutral sentiment.
bar_height = sentiment_score if sentiment_score != 0 else 0.05  # Slight height for neutral

sns.barplot(
    x=["Sentiment Score"],
    y=[bar_height],
    palette=[sentiment_color]
)

plt.ylim(-1, 1)
plt.axhline(0, color="black", linestyle="--")  # Neutral reference line
plt.title("Sentiment Score (-1 Negative → +1 Positive)")
plt.ylabel("Polarity")
plt.show()

# 2. WordCloud from input text
stop_words = set(stopwords.words("english"))
wordcloud = WordCloud(
    width=800, height=400, background_color="white",
    stopwords=stop_words
).generate(user_text)

plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Your Text")
plt.show()
