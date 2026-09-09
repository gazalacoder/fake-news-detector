from flask import Flask, render_template, request, redirect, send_file
import os
import re
import io
import joblib
import nltk

from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# =========================================
# DOWNLOAD NLTK DATA
# =========================================

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
except LookupError:
    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()


# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)


# =========================================
# PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tfidf_vectorizer.pkl"
)

ACCURACY_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model_accuracy.txt"
)


# =========================================
# LOAD MODEL
# =========================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =========================================
# LOAD ACCURACY
# =========================================

accuracy = "Not Available"

if os.path.exists(ACCURACY_PATH):
    with open(ACCURACY_PATH, "r") as file:
        accuracy = file.read().strip()


# =========================================
# HISTORY
# =========================================

history = []


# =========================================
# LATEST RESULT
# =========================================

latest_result = {
    "news": "",
    "prediction": "",
    "confidence": "",
    "date": ""
}


# =========================================
# TEXT PREPROCESSING
# =========================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================================
# HOME PAGE + PREDICTION
# =========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    news_text = ""

    if request.method == "POST":

        news_text = request.form.get(
            "news",
            ""
        )

        if news_text.strip():

            clean_text = preprocess_text(
                news_text
            )

            text_vector = vectorizer.transform(
                [clean_text]
            )

            prediction = model.predict(
                text_vector
            )[0]

            probabilities = model.predict_proba(
                text_vector
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )

            # Convert prediction to REAL or FAKE
            if (
                prediction == 1
                or str(prediction).upper() == "REAL"
            ):
                prediction = "REAL"
            else:
                prediction = "FAKE"

            analyzed_date = datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            )

            # Save latest result
            latest_result["news"] = news_text
            latest_result["prediction"] = prediction
            latest_result["confidence"] = confidence
            latest_result["date"] = analyzed_date

            # Add to history
            history.insert(
                0,
                {
                    "text": (
                        news_text[:100] + "..."
                        if len(news_text) > 100
                        else news_text
                    ),
                    "prediction": prediction,
                    "confidence": confidence,
                    "date": analyzed_date
                }
            )

            # Keep only 5 results
            if len(history) > 5:
                history.pop()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text,
        history=history,
        accuracy=accuracy
    )


# =========================================
# CLEAR HISTORY
# =========================================

@app.route("/clear-history")
def clear_history():

    history.clear()

    return redirect("/")


# =========================================
# DOWNLOAD PDF REPORT
# =========================================

@app.route("/download-report")
def download_report():

    if not latest_result["prediction"]:
        return redirect("/")

    buffer = io.BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=letter
    )

    width, height = letter

    # Title
    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        height - 60,
        "Fake News Detector Report"
    )

    # Details
    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        height - 110,
        f"Prediction: {latest_result['prediction']}"
    )

    pdf.drawString(
        50,
        height - 140,
        f"Confidence: {latest_result['confidence']}%"
    )

    pdf.drawString(
        50,
        height - 170,
        f"Model Accuracy: {accuracy}%"
    )

    pdf.drawString(
        50,
        height - 200,
        f"Analyzed On: {latest_result['date']}"
    )

    # News heading
    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        height - 250,
        "Analyzed News:"
    )

    # News text
    pdf.setFont(
        "Helvetica",
        11
    )

    words = latest_result["news"].split()

    line = ""
    y = height - 280

    for word in words:

        test_line = line + word + " "

        if pdf.stringWidth(
            test_line,
            "Helvetica",
            11
        ) < width - 100:

            line = test_line

        else:

            pdf.drawString(
                50,
                y,
                line
            )

            y -= 20

            line = word + " "

            if y < 70:

                pdf.showPage()

                y = height - 60

                pdf.setFont(
                    "Helvetica",
                    11
                )

    if line:

        pdf.drawString(
            50,
            y,
            line
        )

    # Footer
    pdf.setFont(
        "Helvetica-Oblique",
        9
    )

    pdf.drawString(
        50,
        40,
        "Generated using Machine Learning and NLP."
    )

    pdf.save()

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="fake_news_report.pdf",
        mimetype="application/pdf"
    )


# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )