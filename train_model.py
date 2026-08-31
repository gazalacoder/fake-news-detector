import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLP resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load dataset
data = pd.read_csv("news_dataset.csv")

# NLP setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Apply NLP preprocessing
data["clean_text"] = data["text"].apply(preprocess_text)

# Features and labels
X = data["clean_text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vector, y_train)

# Test model
predictions = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
with open("model/model_accuracy.txt", "w") as file:
    file.write(f"{accuracy * 100:.2f}")


print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")