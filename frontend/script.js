/**
 * Professional Loan Prediction System JavaScript
 * Handles form validation, API communication, and dynamic UI updates
 */

class LoanPredictionSystem {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 4;
        this.formData = {};
        this.validationRules = this.initializeValidationRules();
        this.apiBaseUrl = window.location.origin + '/api';
        this.predictionCount = parseInt(localStorage.getItem('predictionCount') || '0');

        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.bindEventListeners();
        this.updatePredictionCount();
        this.loadModelInfo();
        this.setupFormValidation();
        this.updateProgressBar();

        // Initialize tooltips and smooth scrolling
        this.initializeEnhancements();

        console.log('🚀 Loan Prediction System initialized successfully');
    }

    /**
     * Initialize validation rules for form fields
     */
    initializeValidationRules() {
        return {
            Gender: {
                required: true,
                type: 'select',
                options: ['Male', 'Female']
            },
            Married: {
                required: true,
                type: 'select',
                options: ['Yes', 'No']
            },
            Dependents: {
                required: true,
                type: 'select',
                options: ['0', '1', '2', '3+']
            },
            Education: {
                required: true,
                type: 'select',
                options: ['Graduate', 'Not Graduate']
            },
            Self_Employed: {
                required: true,
                type: 'select',
                options: ['Yes', 'No']
            },
            ApplicantIncome: {
                required: true,
                type: 'number',
                min: 0,
                max: 100000,
                message: 'Please enter a valid monthly income between $0 and $100,000'
            },
            CoapplicantIncome: {
                required: false,
                type: 'number',
                min: 0,
                max: 100000,
                message: 'Please enter a valid co-applicant income between $0 and $100,000'
            },
            LoanAmount: {
                required: true,
                type: 'number',
                min: 1,
                max: 10000,
                message: 'Please enter a valid loan amount between $1,000 and $10,000,000'
            },
            Loan_Amount_Term: {
                required: true,
                type: 'select',
                options: ['120', '180', '240', '300', '360']
            },
            Credit_History: {
                required: true,
                type: 'select',
                options: ['0', '1']
            },
            Property_Area: {
                required: true,
                type: 'select',
                options: ['Urban', 'Semiurban', 'Rural']
            }
        };
    }

    /**
     * Bind all event listeners
     */
    bindEventListeners() {
        // Navigation buttons
        document.getElementById('nextBtn').addEventListener('click', () => this.nextStep());
        document.getElementById('prevBtn').addEventListener('click', () => this.prevStep());
        document.getElementById('submitBtn').addEventListener('click', (e) => this.handleSubmit(e));

        // Form submission
        document.getElementById('loanForm').addEventListener('submit', (e) => this.handleSubmit(e));

        // Real-time validation
        document.querySelectorAll('.form-control').forEach(input => {
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.handleInputChange(input));
        });

        // Financial calculations
        ['applicantIncome', 'coapplicantIncome', 'loanAmount', 'loanTerm'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('input', () => this.updateFinancialSummary());
            }
        });

        // Navigation menu
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => this.handleNavigation(e));
        });

        // Results actions
        const newAppBtn = document.getElementById('newApplicationBtn');
        const downloadBtn = document.getElementById('downloadReportBtn');

        if (newAppBtn) {
            newAppBtn.addEventListener('click', () => this.resetApplication());
        }

        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadReport());
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => this.handleKeyboardNavigation(e));
    }

    /**
     * Initialize additional enhancements
     */
    initializeEnhancements() {
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Auto-save form data
        setInterval(() => this.autoSaveFormData(), 30000); // Every 30 seconds

        // Load saved form data
        this.loadSavedFormData();
    }

    /**
     * Handle keyboard navigation
     */
    handleKeyboardNavigation(e) {
        if (e.key === 'Enter' && e.target.classList.contains('form-control')) {
            e.preventDefault();
            const currentStepElement = document.querySelector('.form-step.active');
            const inputs = currentStepElement.querySelectorAll('.form-control');
            const currentIndex = Array.from(inputs).indexOf(e.target);

            if (currentIndex < inputs.length - 1) {
                inputs[currentIndex + 1].focus();
            } else {
                this.nextStep();
            }
        }
    }

    /**
     * Handle navigation menu clicks
     */
    handleNavigation(e) {
        e.preventDefault();
        const target = e.target.getAttribute('href');

        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
        e.target.classList.add('active');

        // Smooth scroll to section
        const targetElement = document.querySelector(target);
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    /**
     * Setup form validation
     */
    setupFormValidation() {
        document.querySelectorAll('.form-control').forEach(input => {
            input.addEventListener('invalid', (e) => {
                e.preventDefault();
                this.showFieldError(input, 'This field is required');
            });
        });
    }

    /**
     * Validate a single field
     */
    validateField(input) {
        const fieldName = input.name;
        const fieldRules = this.validationRules[fieldName];
        const value = input.value.trim();

        if (!fieldRules) return true;

        // Clear previous errors
        this.clearFieldError(input);

        // Required field validation
        if (fieldRules.required && !value) {
            this.showFieldError(input, 'This field is required');
            return false;
        }

        // Skip further validation if field is empty and not required
        if (!value && !fieldRules.required) return true;

        // Type-specific validation
        if (fieldRules.type === 'number') {
            const numValue = parseFloat(value);

            if (isNaN(numValue)) {
                this.showFieldError(input, 'Please enter a valid number');
                return false;
            }

            if (fieldRules.min !== undefined && numValue < fieldRules.min) {
                this.showFieldError(input, `Value must be at least ${fieldRules.min}`);
                return false;
            }

            if (fieldRules.max !== undefined && numValue > fieldRules.max) {
                this.showFieldError(input, `Value must not exceed ${fieldRules.max}`);
                return false;
            }
        }

        if (fieldRules.type === 'select' && fieldRules.options) {
            if (!fieldRules.options.includes(value)) {
                this.showFieldError(input, 'Please select a valid option');
                return false;
            }
        }

        // Custom validation messages
        if (fieldRules.message && !this.validateFieldValue(fieldName, value)) {
            this.showFieldError(input, fieldRules.message);
            return false;
        }

        // Field passed validation
        this.showFieldSuccess(input);
        return true;
    }

    /**
     * Custom field value validation
     */
    validateFieldValue(fieldName, value) {
        switch (fieldName) {
            case 'ApplicantIncome':
            case 'CoapplicantIncome':
                return !isNaN(value) && parseFloat(value) >= 0 && parseFloat(value) <= 100000;
            case 'LoanAmount':
                return !isNaN(value) && parseFloat(value) >= 1 && parseFloat(value) <= 10000;
            default:
                return true;
        }
    }

    /**
     * Show field error
     */
    showFieldError(input, message) {
        input.classList.add('error');
        input.classList.remove('success');

        const validationDiv = input.parentNode.querySelector('.field-validation');
        if (validationDiv) {
            validationDiv.textContent = message;
            validationDiv.style.color = 'var(--error-color)';
        }
    }

    /**
     * Show field success
     */
    showFieldSuccess(input) {
        input.classList.add('success');
        input.classList.remove('error');

        const validationDiv = input.parentNode.querySelector('.field-validation');
        if (validationDiv) {
            validationDiv.textContent = '✓ Valid';
            validationDiv.style.color = 'var(--success-color)';
        }
    }

    /**
     * Clear field error
     */
    clearFieldError(input) {
        input.classList.remove('error', 'success');

        const validationDiv = input.parentNode.querySelector('.field-validation');
        if (validationDiv) {
            validationDiv.textContent = '';
        }
    }

    /**
     * Handle input changes for real-time updates
     */
    handleInputChange(input) {
        // Real-time validation for immediate feedback
        if (input.value.trim()) {
            setTimeout(() => this.validateField(input), 300);
        } else {
            this.clearFieldError(input);
        }

        // Update financial summary for relevant fields
        if (['applicantIncome', 'coapplicantIncome', 'loanAmount', 'loanTerm'].includes(input.id)) {
            setTimeout(() => this.updateFinancialSummary(), 100);
        }
    }

    /**
     * Update financial summary
     */
    updateFinancialSummary() {
        const applicantIncome = parseFloat(document.getElementById('applicantIncome').value) || 0;
        const coapplicantIncome = parseFloat(document.getElementById('coapplicantIncome').value) || 0;
        const loanAmount = parseFloat(document.getElementById('loanAmount').value) || 0;
        const loanTerm = parseFloat(document.getElementById('loanTerm').value) || 360;

        const totalIncome = applicantIncome + coapplicantIncome;
        const incomeRatio = totalIncome > 0 ? (loanAmount * 1000 / totalIncome).toFixed(1) : '0';
        const monthlyPayment = this.calculateMonthlyPayment(loanAmount * 1000, loanTerm);

        // Update display
        document.getElementById('totalIncome').textContent = `$${totalIncome.toLocaleString()}`;
        document.getElementById('incomeRatio').textContent = `${incomeRatio}:1`;
        document.getElementById('monthlyPayment').textContent = `$${monthlyPayment.toLocaleString()}`;

        // Add visual indicators for risk assessment
        this.updateRiskIndicators(totalIncome, loanAmount * 1000, monthlyPayment);
    }

    /**
     * Calculate monthly payment (simplified)
     */
    calculateMonthlyPayment(loanAmount, termMonths, annualRate = 0.065) {
        if (loanAmount <= 0 || termMonths <= 0) return 0;

        const monthlyRate = annualRate / 12;
        const payment = loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, termMonths)) /
            (Math.pow(1 + monthlyRate, termMonths) - 1);

        return Math.round(payment);
    }

    /**
     * Update risk indicators in financial summary
     */
    updateRiskIndicators(totalIncome, loanAmount, monthlyPayment) {
        const summaryItems = document.querySelectorAll('.summary-item');

        summaryItems.forEach(item => {
            const label = item.querySelector('.summary-label').textContent;
            const value = item.querySelector('.summary-value');

            // Remove existing classes
            value.classList.remove('risk-low', 'risk-medium', 'risk-high');

            if (label.includes('Income-to-Loan Ratio')) {
                const ratio = parseFloat(value.textContent.split(':')[0]);
                if (ratio < 3) value.classList.add('risk-low');
                else if (ratio < 5) value.classList.add('risk-medium');
                else value.classList.add('risk-high');
            }

            if (label.includes('Monthly Payment') && totalIncome > 0) {
                const paymentRatio = monthlyPayment / totalIncome;
                if (paymentRatio < 0.3) value.classList.add('risk-low');
                else if (paymentRatio < 0.4) value.classList.add('risk-medium');
                else value.classList.add('risk-high');
            }
        });
    }

    /**
     * Validate current step
     */
    validateCurrentStep() {
        const currentStepElement = document.querySelector(`.form-step[data-step="${this.currentStep}"]`);
        const inputs = currentStepElement.querySelectorAll('.form-control[required]');
        let isValid = true;

        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });

        return isValid;
    }

    /**
     * Move to next step
     */
    nextStep() {
        if (this.currentStep >= this.totalSteps) return;

        // Validate current step
        if (!this.validateCurrentStep()) {
            this.showToast('Please correct the errors before proceeding', 'error');
            return;
        }

        // Save current step data
        this.saveCurrentStepData();

        // Move to next step
        this.currentStep++;
        this.updateFormStep();
        this.updateProgressBar();
        this.updateNavigationButtons();

        // Show success for completing step
        if (this.currentStep <= 3) {
            this.showToast(`Step ${this.currentStep - 1} completed successfully!`, 'success');
        }
    }

    /**
     * Move to previous step
     */
    prevStep() {
        if (this.currentStep <= 1) return;

        this.currentStep--;
        this.updateFormStep();
        this.updateProgressBar();
        this.updateNavigationButtons();
    }

    /**
     * Update form step visibility
     */
    updateFormStep() {
        // Hide all steps
        document.querySelectorAll('.form-step').forEach(step => {
            step.classList.remove('active');
        });

        // Show current step
        const currentStep = document.querySelector(`.form-step[data-step="${this.currentStep}"]`);
        if (currentStep) {
            currentStep.classList.add('active');

            // Focus first input in current step
            const firstInput = currentStep.querySelector('.form-control');
            if (firstInput) {
                setTimeout(() => firstInput.focus(), 300);
            }
        }

        // Update step indicators
        document.querySelectorAll('.step').forEach((step, index) => {
            const stepNumber = index + 1;
            step.classList.remove('active', 'completed');

            if (stepNumber === this.currentStep) {
                step.classList.add('active');
            } else if (stepNumber < this.currentStep) {
                step.classList.add('completed');
            }
        });
    }

    /**
     * Update progress bar
     */
    updateProgressBar() {
        const progressFill = document.getElementById('progressFill');
        const progressPercent = (this.currentStep / this.totalSteps) * 100;

        if (progressFill) {
            progressFill.style.width = `${progressPercent}%`;
        }
    }

    /**
     * Update navigation buttons
     */
    updateNavigationButtons() {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const submitBtn = document.getElementById('submitBtn');

        // Previous button
        if (prevBtn) {
            prevBtn.style.display = this.currentStep > 1 ? 'inline-flex' : 'none';
        }

        // Next/Submit buttons
        if (this.currentStep < 3) {
            if (nextBtn) nextBtn.style.display = 'inline-flex';
            if (submitBtn) submitBtn.style.display = 'none';
        } else {
            if (nextBtn) nextBtn.style.display = 'none';
            if (submitBtn) submitBtn.style.display = 'inline-flex';
        }
    }

    /**
     * Save current step data
     */
    saveCurrentStepData() {
        const currentStepElement = document.querySelector(`.form-step[data-step="${this.currentStep}"]`);
        const inputs = currentStepElement.querySelectorAll('.form-control');

        inputs.forEach(input => {
            this.formData[input.name] = input.value;
        });

        // Auto-save to localStorage
        this.autoSaveFormData();
    }

    /**
     * Auto-save form data to localStorage
     */
    autoSaveFormData() {
        try {
            localStorage.setItem('loanFormData', JSON.stringify(this.formData));
        } catch (e) {
            console.warn('Could not save form data to localStorage:', e);
        }
    }

    /**
     * Load saved form data from localStorage
     */
    loadSavedFormData() {
        try {
            const savedData = localStorage.getItem('loanFormData');
            if (savedData) {
                const parsedData = JSON.parse(savedData);

                // Populate form fields
                Object.keys(parsedData).forEach(fieldName => {
                    const input = document.querySelector(`[name="${fieldName}"]`);
                    if (input && parsedData[fieldName]) {
                        input.value = parsedData[fieldName];
                        this.formData[fieldName] = parsedData[fieldName];
                    }
                });

                this.updateFinancialSummary();
            }
        } catch (e) {
            console.warn('Could not load saved form data:', e);
        }
    }

    /**
     * Handle form submission
     */
    async handleSubmit(e) {
        e.preventDefault();

        // Final validation
        if (!this.validateAllSteps()) {
            this.showToast('Please complete all required fields correctly', 'error');
            return;
        }

        // Collect all form data
        this.collectFormData();

        // Show loading state
        this.showLoadingState();

        try {
            // Make prediction request
            const prediction = await this.makePrediction(this.formData);

            // Show results
            this.showResults(prediction);

            // Update prediction count
            this.updatePredictionCount(1);

            // Clear saved form data
            localStorage.removeItem('loanFormData');

        } catch (error) {
            console.error('Prediction error:', error);
            this.hideLoadingState();
            this.showToast('Failed to get prediction. Please try again.', 'error');
        }
    }

    /**
     * Validate all form steps
     */
    validateAllSteps() {
        let isValid = true;

        for (let step = 1; step <= 3; step++) {
            const stepElement = document.querySelector(`.form-step[data-step="${step}"]`);
            const inputs = stepElement.querySelectorAll('.form-control[required]');

            inputs.forEach(input => {
                if (!this.validateField(input)) {
                    isValid = false;
                }
            });
        }

        return isValid;
    }

    /**
     * Collect all form data
     */
    collectFormData() {
        const formElement = document.getElementById('loanForm');
        const formData = new FormData(formElement);

        this.formData = {};
        for (let [key, value] of formData.entries()) {
            this.formData[key] = value;
        }

        // Ensure numeric fields are properly typed
        const numericFields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'];
        numericFields.forEach(field => {
            if (this.formData[field]) {
                this.formData[field] = parseFloat(this.formData[field]);
            }
        });

        console.log('📋 Form data collected:', this.formData);
    }

    /**
     * Make prediction API call
     */
    async makePrediction(data) {
        const response = await fetch(`${this.apiBaseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Show loading state
     */
    showLoadingState() {
        // Scroll to results section
        document.getElementById('results').style.display = 'block';
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

        // Show loading overlay
        document.getElementById('loadingOverlay').style.display = 'flex';
        document.getElementById('resultsContent').style.display = 'none';

        // Disable form
        document.getElementById('loanForm').style.pointerEvents = 'none';
        document.getElementById('submitBtn').disabled = true;
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'grid';

        // Re-enable form
        document.getElementById('loanForm').style.pointerEvents = 'auto';
        document.getElementById('submitBtn').disabled = false;
    }

    /**
     * Show prediction results
     */
    showResults(response) {
        this.hideLoadingState();

        const prediction = response.prediction;
        const isApproved = prediction.prediction === 1;

        // Update prediction result
        const predictionIcon = document.getElementById('predictionIcon');
        const predictionResult = document.getElementById('predictionResult');
        const predictionSubtext = document.getElementById('predictionSubtext');
        const predictionHeader = predictionIcon.closest('.prediction-header');

        if (isApproved) {
            predictionIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
            predictionResult.textContent = 'Loan Approved';
            predictionSubtext.textContent = 'Congratulations! Your application shows positive indicators.';
            predictionHeader.classList.remove('rejected');
        } else {
            predictionIcon.innerHTML = '<i class="fas fa-times-circle"></i>';
            predictionResult.textContent = 'Loan Rejected';
            predictionSubtext.textContent = 'Your application requires further review or improvement.';
            predictionHeader.classList.add('rejected');
        }

        // Update confidence meter
        const confidence = prediction.confidence || 0.5;
        const confidencePercent = Math.round(confidence * 100);

        document.getElementById('confidencePercent').textContent = `${confidencePercent}%`;
        document.getElementById('confidenceFill').style.width = `${confidencePercent}%`;

        // Update probability breakdown
        if (prediction.probabilities) {
            document.getElementById('approvalProb').textContent =
                `${Math.round(prediction.probabilities.approved * 100)}%`;
            document.getElementById('rejectionProb').textContent =
                `${Math.round(prediction.probabilities.rejected * 100)}%`;
        }

        // Update model information
        const modelInfo = response.model_info;
        if (modelInfo) {
            document.getElementById('modelType').textContent =
                modelInfo.model_type?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Decision Tree';
            document.getElementById('modelAccuracy').textContent =
                modelInfo.accuracy ? `${Math.round(modelInfo.accuracy * 100)}%` : 'N/A';
        }

        // Show key factors
        this.showKeyFactors(prediction, response.input_data);

        // Add animations
        document.getElementById('resultsContent').classList.add('fade-in');

        this.showToast('Prediction completed successfully!', 'success');
    }

    /**
     * Show key decision factors
     */
    showKeyFactors(prediction, inputData) {
        const factorsList = document.getElementById('factorsList');
        factorsList.innerHTML = '';

        // Define factor importance and descriptions
        const factorDescriptions = {
            'Credit_History': {
                icon: 'fas fa-history',
                title: 'Credit History',
                getDescription: (value) => value === '1' ? 'Good credit history strengthens your application' : 'Poor credit history may impact approval',
                getImpact: (value) => value === '1' ? 'Positive' : 'Negative'
            },
            'ApplicantIncome': {
                icon: 'fas fa-dollar-sign',
                title: 'Applicant Income',
                getDescription: (value) => `Monthly income of $${parseFloat(value).toLocaleString()} considered`,
                getImpact: (value) => parseFloat(value) > 5000 ? 'Positive' : 'Moderate'
            },
            'LoanAmount': {
                icon: 'fas fa-money-check',
                title: 'Loan Amount',
                getDescription: (value) => `Requested amount: $${parseFloat(value * 1000).toLocaleString()}`,
                getImpact: (value) => parseFloat(value) < 200 ? 'Positive' : 'Moderate'
            },
            'Education': {
                icon: 'fas fa-graduation-cap',
                title: 'Education Level',
                getDescription: (value) => `${value} status affects risk assessment`,
                getImpact: (value) => value === 'Graduate' ? 'Positive' : 'Neutral'
            },
            'Property_Area': {
                icon: 'fas fa-map-marker-alt',
                title: 'Property Location',
                getDescription: (value) => `${value} area property affects valuation`,
                getImpact: (value) => value === 'Urban' ? 'Positive' : 'Neutral'
            }
        };

        // Create factor items for key fields
        const keyFactors = ['Credit_History', 'ApplicantIncome', 'LoanAmount', 'Education', 'Property_Area'];

        keyFactors.forEach(factorKey => {
            if (inputData[factorKey] !== undefined && factorDescriptions[factorKey]) {
                const factor = factorDescriptions[factorKey];
                const value = inputData[factorKey];

                const factorElement = document.createElement('div');
                factorElement.className = 'factor-item';
                factorElement.innerHTML = `
                    <div class="factor-icon">
                        <i class="${factor.icon}"></i>
                    </div>
                    <div class="factor-content">
                        <div class="factor-title">${factor.title}</div>
                        <div class="factor-description">${factor.getDescription(value)}</div>
                    </div>
                    <div class="factor-impact">${factor.getImpact(value)}</div>
                `;

                factorsList.appendChild(factorElement);
            }
        });
    }

    /**
     * Load model information from API
     */
    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model-info`);
            if (response.ok) {
                const data = await response.json();
                const modelInfo = data.model_info;

                // Update hero stats
                if (modelInfo.training_metrics?.accuracy) {
                    document.getElementById('model-accuracy').textContent =
                        `${Math.round(modelInfo.training_metrics.accuracy * 100)}%`;
                }

                // Update analytics section
                this.updateAnalyticsSection(modelInfo);
            }
        } catch (error) {
            console.warn('Could not load model info:', error);
        }
    }

    /**
     * Update analytics section with model data
     */
    updateAnalyticsSection(modelInfo) {
        // Update performance metrics
        if (modelInfo.training_metrics) {
            const metrics = modelInfo.training_metrics;

            document.getElementById('overallAccuracy').textContent =
                metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A';
            document.getElementById('modelPrecision').textContent =
                metrics.precision ? `${(metrics.precision * 100).toFixed(1)}%` : 'N/A';
            document.getElementById('modelRecall').textContent =
                metrics.recall ? `${(metrics.recall * 100).toFixed(1)}%` : 'N/A';
        }

        // Update feature importance
        if (modelInfo.feature_importance) {
            this.updateFeatureImportance(modelInfo.feature_importance);
        }

        // Update feature count
        if (modelInfo.feature_count) {
            document.getElementById('featureCount').textContent = modelInfo.feature_count;
        }

        // Update system status
        document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
    }

    /**
     * Update feature importance display
     */
    updateFeatureImportance(featureImportance) {
        const featureList = document.getElementById('featureImportanceList');
        if (!featureList) return;

        featureList.innerHTML = '';

        // Convert object to array and sort by importance
        const sortedFeatures = Object.entries(featureImportance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8); // Top 8 features

        sortedFeatures.forEach(([featureName, importance]) => {
            const featureElement = document.createElement('div');
            featureElement.className = 'feature-item';
            featureElement.innerHTML = `
                <span class="feature-name">${this.formatFeatureName(featureName)}</span>
                <span class="feature-score">${(importance * 100).toFixed(1)}%</span>
            `;
            featureList.appendChild(featureElement);
        });
    }

    /**
     * Format feature name for display
     */
    formatFeatureName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace('Coapplicant', 'Co-applicant');
    }

    /**
     * Update prediction count
     */
    updatePredictionCount(increment = 0) {
        this.predictionCount += increment;
        localStorage.setItem('predictionCount', this.predictionCount.toString());

        const countElement = document.getElementById('predictions-made');
        if (countElement) {
            countElement.textContent = this.predictionCount.toLocaleString();
        }
    }

    /**
     * Reset application for new prediction
     */
    resetApplication() {
        // Reset form
        document.getElementById('loanForm').reset();
        this.formData = {};

        // Reset to first step
        this.currentStep = 1;
        this.updateFormStep();
        this.updateProgressBar();
        this.updateNavigationButtons();

        // Clear validation states
        document.querySelectorAll('.form-control').forEach(input => {
            this.clearFieldError(input);
        });

        // Hide results
        document.getElementById('results').style.display = 'none';

        // Clear saved data
        localStorage.removeItem('loanFormData');

        // Scroll to form
        document.getElementById('home').scrollIntoView({ behavior: 'smooth' });

        this.showToast('Ready for new application', 'success');
    }

    /**
     * Download prediction report
     */
    downloadReport() {
        // Create a simple report (in a real application, you might generate a PDF)
        const reportContent = this.generateReportContent();

        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `loan_prediction_report_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();

        URL.revokeObjectURL(url);

        this.showToast('Report downloaded successfully', 'success');
    }

    /**
     * Generate report content
     */
    generateReportContent() {
        const timestamp = new Date().toLocaleString();
        const predictionResult = document.getElementById('predictionResult').textContent;
        const confidence = document.getElementById('confidencePercent').textContent;

        return `
LOAN PREDICTION REPORT
Generated: ${timestamp}

PREDICTION RESULT: ${predictionResult}
CONFIDENCE LEVEL: ${confidence}

APPLICATION DETAILS:
${Object.entries(this.formData).map(([key, value]) => `${key}: ${value}`).join('\n')}

DISCLAIMER:
This prediction is generated by an AI model and should be used for informational purposes only.
Actual loan decisions should be made by qualified financial professionals.

Generated by LoanPredict Pro - Professional Loan Prediction System
        `.trim();
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 5000) {
        const toastContainer = document.getElementById('toastContainer');

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const iconClass = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle',
            warning: 'fas fa-exclamation-circle'
        }[type] || 'fas fa-info-circle';

        toast.innerHTML = `
            <div class="toast-content">
                <i class="toast-icon ${iconClass}"></i>
                <div class="toast-text">
                    <div class="toast-title">${type.charAt(0).toUpperCase() + type.slice(1)}</div>
                    <div class="toast-message">${message}</div>
                </div>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Auto-remove toast
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toastContainer.removeChild(toast), 300);
        }, duration);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LoanPredictionSystem();
});

// Add some CSS for risk indicators and toast animations
const additionalStyles = `
<style>
.risk-low { color: var(--success-color) !important; }
.risk-medium { color: var(--warning-color) !important; }
.risk-high { color: var(--error-color) !important; }

.form-control.error {
    border-color: var(--error-color) !important;
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1) !important;
}

.form-control.success {
    border-color: var(--success-color) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
}

.toast {
    transform: translateX(0);
    transition: all 0.3s ease;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', additionalStyles);