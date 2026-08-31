from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -----------------------------
# NLTK STOPWORDS SETUP
# -----------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

# -----------------------------
# PATHS
# -----------------------------
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

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Prediction History
prediction_history = []


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICT NEWS
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "news" not in data:
        return jsonify({
            "error": "Please enter news text."
        })

    news = data["news"].strip()

    if not news:
        return jsonify({
            "error": "Please enter news text."
        })

    # Clean text
    cleaned_news = clean_text(news)

    # Convert into TF-IDF
    news_vector = vectorizer.transform([cleaned_news])

    # Prediction
    prediction = model.predict(news_vector)[0]

    # Probability / Confidence
    probability = model.predict_proba(news_vector)[0]

    confidence = round(max(probability) * 100, 2)

    # Handle different dataset labels
    if prediction == 1 or str(prediction).lower() == "real":
        result = "REAL"
    else:
        result = "FAKE"

    # Date and Time
    current_time = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    # Save history
    history_item = {
        "news": news,
        "result": result,
        "confidence": confidence,
        "time": current_time
    }

    prediction_history.insert(0, history_item)

    # Keep only last 20 predictions
    if len(prediction_history) > 20:
        prediction_history.pop()

    return jsonify({
        "result": result,
        "confidence": confidence,
        "time": current_time
    })


# -----------------------------
# GET HISTORY
# -----------------------------
@app.route("/history")
def history():

    return jsonify(prediction_history)


# -----------------------------
# CLEAR HISTORY
# -----------------------------
@app.route("/clear-history", methods=["POST"])
def clear_history():

    prediction_history.clear()

    return jsonify({
        "message": "History cleared successfully."
    })


# -----------------------------
# PDF REPORT
# -----------------------------
@app.route("/download-report", methods=["POST"])
def download_report():

    data = request.get_json()

    news = data.get("news", "")
    result = data.get("result", "")
    confidence = data.get("confidence", "")
    time = data.get(
        "time",
        datetime.now().strftime("%d %B %Y, %I:%M %p")
    )

    report_path = os.path.join(
        BASE_DIR,
        "fake_news_report.pdf"
    )

    pdf = canvas.Canvas(
        report_path,
        pagesize=A4
    )

    width, height = A4

    # Title
    pdf.setFont("Helvetica-Bold", 20)

    pdf.drawString(
        50,
        height - 60,
        "Fake News Detection Report"
    )

    # Date
    pdf.setFont("Helvetica", 11)

    pdf.drawString(
        50,
        height - 100,
        f"Date & Time: {time}"
    )

    # Result
    pdf.setFont("Helvetica-Bold", 14)

    pdf.drawString(
        50,
        height - 140,
        f"Prediction: {result}"
    )

    pdf.drawString(
        50,
        height - 170,
        f"Confidence: {confidence}%"
    )

    # News Text
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        50,
        height - 220,
        "News Text:"
    )

    pdf.setFont("Helvetica", 10)

    # Simple text wrapping
    words = news.split()

    line = ""
    y_position = height - 250

    for word in words:

        test_line = line + word + " "

        if pdf.stringWidth(
            test_line,
            "Helvetica",
            10
        ) < 500:

            line = test_line

        else:

            pdf.drawString(
                50,
                y_position,
                line
            )

            y_position -= 20

            line = word + " "

            # New page if needed
            if y_position < 60:

                pdf.showPage()

                y_position = height - 60

                pdf.setFont(
                    "Helvetica",
                    10
                )

    if line:

        pdf.drawString(
            50,
            y_position,
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
        "Generated by Fake News Detector"
    )

    pdf.save()

    return send_file(
        report_path,
        as_attachment=True,
        download_name="fake_news_report.pdf"
    )


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )