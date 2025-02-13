document.getElementById("loginForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;

    if (email === "" || password === "") {
        alert("All fields are required!");
        return;
    }

    // Dummy check (In real application, validate with backend database)
    if (email === "test@example.com" && password === "password123") {
        alert("Login successful!");
        window.location.href = "dashboard.html"; // Redirect to dashboard
    } else {
        alert("Invalid email or password!");
    }
});
