from flask import Flask, render_template, request, redirect, send_file
import joblib
import re
import os
import io

from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


app = Flask(__name__)


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = joblib.load("model/fake_news_model.pkl")

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


# =========================================
# LOAD MODEL ACCURACY
# =========================================

accuracy = "Not Available"

accuracy_file = "model/model_accuracy.txt"

if os.path.exists(accuracy_file):

    with open(accuracy_file, "r") as file:

        accuracy = file.read().strip()


# =========================================
# PREDICTION HISTORY
# =========================================

history = []


# =========================================
# LATEST RESULT FOR PDF REPORT
# =========================================

latest_result = {
    "news": "",
    "prediction": "",
    "confidence": "",
    "date": ""
}


# =========================================
# NLP SETUP
# =========================================

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


# =========================================
# TEXT PREPROCESSING
# =========================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Split text into words
    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [

        lemmatizer.lemmatize(word)

        for word in words

        if word not in stop_words

    ]

    return " ".join(words)


# =========================================
# HOME PAGE
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

            # Clean news text
            clean_text = preprocess_text(
                news_text
            )

            # Convert text into TF-IDF
            text_vector = vectorizer.transform(
                [clean_text]
            )

            # Make prediction
            prediction = model.predict(
                text_vector
            )[0]

            # Calculate confidence
            probabilities = model.predict_proba(
                text_vector
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )

            # Current date and time
            analyzed_date = datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            )

            # Save latest prediction
            latest_result["news"] = news_text

            latest_result["prediction"] = prediction

            latest_result["confidence"] = confidence

            latest_result["date"] = analyzed_date


            # Add prediction to history
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


            # Keep only last 5 predictions
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

    # If no prediction exists
    if not latest_result["prediction"]:

        return redirect("/")


    # Create PDF in memory
    buffer = io.BytesIO()


    pdf = canvas.Canvas(

        buffer,

        pagesize=letter

    )


    width, height = letter


    # =====================================
    # PDF TITLE
    # =====================================

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(

        50,

        height - 60,

        "Fake News Detector Report"

    )


    # =====================================
    # REPORT DETAILS
    # =====================================

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


    # =====================================
    # ANALYZED NEWS
    # =====================================

    pdf.setFont(
        "Helvetica-Bold",
        12
    )


    pdf.drawString(

        50,

        height - 250,

        "Analyzed News:"

    )


    pdf.setFont(
        "Helvetica",
        11
    )


    news = latest_result["news"]

    words = news.split()

    line = ""

    y = height - 280


    # Split long news into multiple lines
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


            # Create new page if needed
            if y < 70:

                pdf.showPage()

                y = height - 60

                pdf.setFont(
                    "Helvetica",
                    11
                )


    # Print remaining text
    if line:

        pdf.drawString(

            50,

            y,

            line

        )


    # =====================================
    # FOOTER
    # =====================================

    pdf.setFont(

        "Helvetica-Oblique",

        9

    )


    pdf.drawString(

        50,

        40,

        "Generated using Machine Learning and NLP."

    )


    # Save PDF
    pdf.save()


    buffer.seek(0)


    # Send PDF to user
    return send_file(

        buffer,

        as_attachment=True,

        download_name="fake_news_report.pdf",

        mimetype="application/pdf"

    )


# =========================================
# RUN APPLICATION
# =========================================

if __name__ == "__main__":

    app.run(debug=True)