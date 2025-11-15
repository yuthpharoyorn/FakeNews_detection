document.addEventListener('DOMContentLoaded', () => {
    // 1. Get the new textarea element
    const articleInput = document.getElementById('article-text'); // Changed from 'link-input'
    const analyzeButton = document.getElementById('analyze-button');
    const statusContainer = document.getElementById('status-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultDisplay = document.getElementById('result-display');

    analyzeButton.addEventListener('click', async () => {
        // 2. Get the text from the textarea
        const articleText = articleInput.value.trim();

        // 3. Basic Validation
        if (!articleText) {
            alert('Please paste an article to analyze.');
            return;
        }

        // 4. Set UI to Loading State (Your code is perfect)
        resetStatusClasses();
        statusContainer.classList.remove('hidden');
        loadingIndicator.classList.remove('hidden');
        resultDisplay.innerHTML = '';

        // 5. --- THIS IS THE NEW PART: Call your real backend API ---
        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: articleText // Send the text in the format the API expects
                }),
            });

            if (!response.ok) {
                // Handle HTTP errors (e.g., 500 server error)
                throw new Error(`API error: ${response.statusText}`);
            }

            const data = await response.json();

            // 6. Update UI based on the *real* API result
            updateResultDisplay(data);

        } catch (error) {
            console.error('Analysis failed:', error);
            // Call your existing error function, but with a more specific message
            updateResultDisplay({ label: 'error', message: 'Could not connect to the API server.' });
        } finally {
            // 7. Hide Loading State (Your code is perfect)
            loadingIndicator.classList.add('hidden');
        }
    });

    /** Resets the status container classes */
    function resetStatusClasses() {
        statusContainer.classList.remove('fake-result', 'real-result', 'error-result');
    }

    /** * Updates the UI with the final result.
     * This is now driven by the API response object 'data'.
     */
    function updateResultDisplay(data) {
        resetStatusClasses();
        
        if (data.label === 'Fake News') {
            statusContainer.classList.add('fake-result');
            const prob = (1 - data.probability_real) * 100;
            resultDisplay.innerHTML = `🚨 **RESULT: FAKE NEWS** 🚨 <br> (${prob.toFixed(1)}% Fake)`;
        } else if (data.label === 'Real News') {
            statusContainer.classList.add('real-result');
            const prob = data.probability_real * 100;
            resultDisplay.innerHTML = `✅ **RESULT: GENUINE** ✅ <br> (${prob.toFixed(1)}% Real)`;
        } else {
            // This handles the 'error' case
            statusContainer.classList.add('error-result');
            resultDisplay.innerHTML = `⚠️ **ANALYSIS FAILED** ⚠️ <br> ${data.message || 'An unknown error occurred.'}`;
        }
    }
});