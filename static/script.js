function clearText() {
    document.getElementById("news").value = "";
}

const themeToggle = document.getElementById("themeToggle");

if (themeToggle) {
    themeToggle.addEventListener("click", function () {
        document.body.classList.toggle("dark-mode");

        if (document.body.classList.contains("dark-mode")) {
            themeToggle.innerHTML = "☀️ Light Mode";
        } else {
            themeToggle.innerHTML = "🌙 Dark Mode";
        }
    });
}


// Confidence Progress Bar
const progressFill = document.querySelector(".progress-fill");

if (progressFill) {
    const confidence = progressFill.getAttribute("data-confidence");

    setTimeout(function () {
        progressFill.style.width = confidence + "%";
    }, 100);
}


// Loading Effect
const newsForm = document.getElementById("newsForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const newsInput = document.getElementById("news");

if (newsForm && analyzeBtn && newsInput) {
    newsForm.addEventListener("submit", function (event) {

        if (newsInput.value.trim() === "") {
            event.preventDefault();
            alert("Please enter some news text first.");
            return;
        }

        analyzeBtn.innerHTML = "⏳ Analyzing...";
        analyzeBtn.disabled = true;
    });
}