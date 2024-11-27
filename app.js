document.getElementById('predictBtn').addEventListener('click', function() {
    // Simulate prediction and change image and text
    const result = "Kidney Stone Detected"; // This should come from your prediction model
    
    document.getElementById('predictionResult').innerText = "Prediction: " + result;
    document.getElementById('resultImage').src('img/1.png');
});

document.getElementById('aboutBtn').addEventListener('click', function() {
    alert("Kidney Stone Detection Model\nBuilt with Streamlit\nBy Your Name");
});
