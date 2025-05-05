// Initialize highlight.js
document.addEventListener('DOMContentLoaded', () => {
    hljs.highlightAll();
});

// const BACKEND_URL = ''; // Update to BACKEND URL after deployment
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const preprocessedPreview = document.getElementById('preprocessedPreview');
const preprocessedLabel = document.getElementById('preprocessedLabel');
const originalLabel = document.getElementById('originalLabel');
const fileName = document.getElementById('fileName');
const submitButton = document.getElementById('submitButton');
const spinner = document.getElementById('spinner');
const statusMessage = document.getElementById('statusMessage');
const predictionResult = document.getElementById('predictionResult');
const llmAdvice = document.getElementById('llmAdvice');
const downloadButton = document.getElementById('downloadButton');

// Handle drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
});
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file && file.type.match('image.*')) {
        fileName.textContent = file.name;
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            originalLabel.style.display = 'block';
            submitButton.classList.remove('hidden');
            fileName.style.display = 'none';
            preprocessedPreview.style.display = 'none';
            preprocessedLabel.style.display = 'none';
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select a valid image file.');
    }
}

submitButton.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) {
        alert('No image selected.');
        return;
    }
    analyzeImage(file);
});

async function analyzeImage(file) {
    fileName.style.display = 'block';
    spinner.classList.add('show');
    submitButton.classList.add('hidden');
    predictionResult.classList.add('hidden');
    llmAdvice.innerHTML = '';
    downloadButton.classList.add('hidden');

    const statusMessages = [
        'Analyzing image...',
        'Preprocessing completed...',
        'Feeding to model...',
        'Fetching recommendations...'
    ];
    let statusIndex = 0;
    statusMessage.textContent = statusMessages[statusIndex];
    const statusInterval = setInterval(() => {
        statusIndex = (statusIndex + 1) % statusMessages.length;
        statusMessage.textContent = statusMessages[statusIndex];
    }, 2000);

    try {
        // Step 1: Preprocess image
        const preprocessFormData = new FormData();
        preprocessFormData.append('file', file);
        const preprocessResponse = await fetch('/preprocess', {
        //const preprocessResponse = await fetch(`${BACKEND_URL}/preprocess`, {
            method: 'POST',
            body: preprocessFormData
        });
        const preprocessData = await preprocessResponse.json();
        if (preprocessData.error) {
            throw new Error(preprocessData.error);
        }
        preprocessedPreview.src = preprocessData.preprocessed_image;
        preprocessedPreview.style.display = 'block';
        preprocessedLabel.style.display = 'block';

        // Step 2: Predict
        const predictFormData = new FormData();
        predictFormData.append('file', file);
        const predictResponse = await fetch('/predict', {
        //const predictResponse = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            body: predictFormData
        });
        const data = await predictResponse.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"];
        predictionResult.textContent = `Diagnosis: ${levels[data.prediction]} (Level ${data.prediction})`;
        predictionResult.className = `result-card level-${data.prediction}`;
        predictionResult.classList.remove('hidden');

        // Step 3: Stream LLM advice
        let markdownText = '';
        let lastRenderTime = 0;
        const renderInterval = 100; // Render every 100ms
        try {
            const streamResponse = await fetch('/stream_advice', {
            //const streamResponse = await fetch(`${BACKEND_URL}/stream_advice`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prediction: data.prediction })
            });
            if (!streamResponse.ok) {
                throw new Error(`HTTP error! status: ${streamResponse.status}`);
            }

            const reader = streamResponse.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    clearInterval(statusInterval);
                    spinner.classList.remove('show');
                    try {
                        console.log("Final Markdown text:", markdownText);
                        llmAdvice.innerHTML = marked.parse(markdownText);
                        hljs.highlightAll();
                    } catch (err) {
                        console.error("Final Markdown parsing error:", err);
                        llmAdvice.innerHTML = "Error rendering recommendations";
                    }
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n\n");

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6);
                        console.log("Raw SSE data:", data);
                        if (data === "[DONE]") {
                            clearInterval(statusInterval);
                            spinner.classList.remove('show');
                            try {
                                console.log("Final Markdown text on [DONE]:", markdownText);
                                llmAdvice.innerHTML = marked.parse(markdownText);
                                hljs.highlightAll();
                            } catch (err) {
                                console.error("Final Markdown parsing error on [DONE]:", err);
                                llmAdvice.innerHTML = "Error rendering recommendations";
                            }
                            break;
                        }
                        try {
                            const json = JSON.parse(data);
                            if (json.error) {
                                throw new Error(json.error);
                            }
                            const content = json.choices?.[0]?.delta?.content || "";
                            if (content) {
                                markdownText += content;
                                console.log("Accumulated Markdown:", markdownText);
                                const now = Date.now();
                                if (now - lastRenderTime >= renderInterval) {
                                    try {
                                        llmAdvice.innerHTML = marked.parse(markdownText);
                                        hljs.highlightAll();
                                        lastRenderTime = now;
                                    } catch (err) {
                                        console.warn("Incremental Markdown parsing error:", err);
                                    }
                                }
                            }
                        } catch (err) {
                            console.error("JSON parse error:", err, "Data:", data);
                        }
                    }
                }
            }
        } catch (err) {
            clearInterval(statusInterval);
            spinner.classList.remove('show');
            throw new Error("Streaming failed: " + err.message);
        }

        // Step 4: Enable download button
        downloadButton.classList.remove('hidden');
        downloadButton.onclick = async () => {
            try {
                const reportData = {
                    original_image: data.original_image,
                    preprocessed_image: data.preprocessed_image,
                    prediction: data.prediction,
                    advice: markdownText,
                    filename: data.filename
                };
                let attempts = 0;
                const maxAttempts = 3;
                while (attempts < maxAttempts) {
                    attempts++;
                    try {
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), 60000);
                        const response = await fetch('/generate_report', {
                        //const response = await fetch(`${BACKEND_URL}/generate_report`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(reportData),
                            signal: controller.signal
                        });
                        clearTimeout(timeoutId);
                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.error || 'Failed to generate report');
                        }
                        const blob = await response.blob();
                        console.log('Attempt', attempts, 'Blob size:', blob.size);
                        if (blob.size < 1000) {
                            throw new Error('Received empty or invalid PDF file');
                        }
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = data.filename;
                        a.click();
                        window.URL.revokeObjectURL(url);
                        return;
                    } catch (error) {
                        if (attempts === maxAttempts) {
                            throw error;
                        }
                        console.log('Retry attempt', attempts + 1, 'after error:', error);
                        await new Promise(resolve => setTimeout(resolve, 2000));
                    }
                }
            } catch (error) {
                console.error('Download error:', error);
                alert('Could not download PDF: ' + (error.message || 'Network error'));
            }
        };
    } catch (error) {
        clearInterval(statusInterval);
        spinner.classList.remove('show');
        submitButton.classList.remove('hidden');
        alert('Analysis failed: ' + error.message);
    }
}
