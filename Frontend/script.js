document.addEventListener('DOMContentLoaded', () => {
    const linkInput = document.getElementById('link-input');
    const analyzeButton = document.getElementById('analyze-button');
    const statusContainer = document.getElementById('status-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultDisplay = document.getElementById('result-display');

    analyzeButton.addEventListener('click', async () => {
        const link = linkInput.value.trim();

     // 1. Basic Validation
        
        if (!link) {
         alert('Please enter a link to analyze.');
            return;
}

// 2. we r using NLP to make input validation
// This pattern checks for: (protocol OR www.) AND (domain structure)
        const urlPattern = /^(https?:\/\/.+|www\..+\..+)/i;

        if (!urlPattern.test(link)) {
            alert('⚠️ Invalid link format. Please ensure you have copied the full URL, including "https://" or "www.".');
            return;
}

        // 2. Set UI to Loading State
        resetStatusClasses();
        statusContainer.classList.remove('hidden');
        loadingIndicator.classList.remove('hidden');
        resultDisplay.innerHTML = ''; // Clear previous result text

        // 3. Simulate interacting with backend, we will update this section once we setup our backend endpoint
        try {
            // --- This block simulates the API call and result ---
           
            await new Promise(resolve => setTimeout(resolve, 3000)); 

            // Simulate 3 possible results (Fake, Real, or API Error)
            const randomChance = Math.random();
            let resultType;

            if (randomChance < 0.45) {
                resultType = 'fake'; 
            } else if (randomChance < 0.9) {
                resultType = 'real'; 
            } else {
                
                resultType = 'error';
            }
           

            // 4. Update UI based on Simulated Result
            updateResultDisplay(resultType);

        } catch (error) {
            console.error('Analysis failed:', error);
            updateResultDisplay('error');
        } finally {
            // 5. Hide Loading State
            loadingIndicator.classList.add('hidden');
        }
    });

    /**reset the status */
    function resetStatusClasses() {
        statusContainer.classList.remove('fake-result', 'real-result', 'error-result');
    }

    /** Updates the UI with the final result */
    function updateResultDisplay(type) {
        resetStatusClasses();
        
        if (type === 'fake') {
            statusContainer.classList.add('fake-result');
            resultDisplay.innerHTML = '🚨 **RESULT: FAKE NEWS** 🚨 <br> This source appears highly questionable.';
        } else if (type === 'real') {
            statusContainer.classList.add('real-result');
            resultDisplay.innerHTML = '✅ **RESULT: GENUINE** ✅ <br> This source appears credible.';
        } else {
            statusContainer.classList.add('error-result');
            resultDisplay.innerHTML = '⚠️ **ANALYSIS FAILED** ⚠️ <br> Could not reach the detector. Check the link and try again.';
        }
    }
});