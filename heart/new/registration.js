document.getElementById("registrationForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let name = document.getElementById("name").value;
    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;
    let age = document.getElementById("age").value;
    let gender = document.getElementById("gender").value;

    if (name === "" || email === "" || password === "" || age === "" || gender === "") {
        alert("All fields are required!");
        return;
    }

    if (password.length < 6) {
        alert("Password must be at least 6 characters long.");
        return;
    }

    alert("Registration successful!");
    window.location.href = "home.html"; // Redirect after successful registration
});
