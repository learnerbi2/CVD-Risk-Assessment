document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let age = document.getElementById("age").value;
    let bp = document.getElementById("bp").value;
    let cholesterol = document.getElementById("cholesterol").value;
    let heartRate = document.getElementById("heartRate").value;
    let glucose = document.getElementById("glucose").value;

    if (age === "" || bp === "" || cholesterol === "" || heartRate === "" || glucose === "") {
        alert("All fields are required!");
        return;
    }

    // Simple Dummy Prediction Logic (Replace with Backend ML Model)
    let risk = (bp > 140 || cholesterol > 200 || heartRate > 100) ? "High Risk of Heart Disease!" : "Low Risk of Heart Disease!";

    document.getElementById("predictionResult").textContent = risk;
    document.getElementById("predictionResult").style.display = "block";
});
