function uploadFile() {
    const fileInput = document.getElementById('inputFile');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image or video.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Send file to backend for classification
    fetch('http://127.0.0.1:5000/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').innerText = `Prediction: ${data.prediction}`;
        document.getElementById('explainability').src = data.explainability_map_url;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to classify file.');
    });
}

document.querySelector('#uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');
    
    formData.append('file', fileField.files[0]);

    const response = await fetch('/classify', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    // Display the classification result
    document.querySelector('#prediction').innerText = `Prediction: ${result.prediction}`;
    
    // Display the explainability map
    document.querySelector('#explainabilityMap').src = result.explainability_map_url;
});
