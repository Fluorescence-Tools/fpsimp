// UI module for handling ColabFold visibility based on configuration

/**
 * Hide ColabFold-related UI elements when ColabFold is disabled
 */
export function hideColabFoldElements() {
    const elements = [
        'colabfoldSection',           // Entire ColabFold section
        'multimerSequenceDisplay',    // Multimer sequence display
        'fastaUploadSection',         // FASTA file upload section
        'pipelineSteps',              // Pipeline steps with ColabFold
    ];

    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'none';
            console.log(`Hidden element: ${id}`);
        }
    });

    // Show disabled versions
    const disabledElements = [
        'pipelineStepsDisabled',       // Pipeline steps without ColabFold
    ];

    disabledElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = '';
            console.log(`Shown element: ${id}`);
        }
    });

    // Show disabled text for PDB upload
    const pdbText = document.getElementById('pdbUploadText');
    const pdbTextDisabled = document.getElementById('pdbUploadTextDisabled');
    if (pdbText) pdbText.style.display = 'none';
    if (pdbTextDisabled) pdbTextDisabled.style.display = '';

    // Update fetch help text and input placeholder
    const fetchHelpText = document.getElementById('fetchHelpText');
    const fetchHelpTextDisabled = document.getElementById('fetchHelpTextDisabled');
    const pdbIdInput = document.getElementById('pdbOrIdInput');
    const fetchBtn = document.getElementById('fetchStructureBtn');

    if (fetchHelpText) fetchHelpText.style.display = 'none';
    if (fetchHelpTextDisabled) fetchHelpTextDisabled.style.display = '';

    // Disable fetch and make input readonly (only for file display)
    if (fetchBtn) fetchBtn.style.display = 'none';
    if (pdbIdInput) {
        pdbIdInput.placeholder = 'Click Browse to upload structure...';
        pdbIdInput.readOnly = true;
    }
}

/**
 * Show ColabFold-related UI elements when ColabFold is enabled
 */
export function showColabFoldElements() {
    const elements = [
        'colabfoldSection',           // Entire ColabFold section
        'multimerSequenceDisplay',    // Multimer sequence display (will be shown/hidden by logic)
        'fastaUploadSection',         // FASTA file upload section
        'pipelineSteps',              // Pipeline steps with ColabFold
    ];

    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = '';
            console.log(`Shown element: ${id}`);
        }
    });

    // Hide disabled versions
    const disabledElements = [
        'pipelineStepsDisabled',       // Pipeline steps without ColabFold
    ];

    disabledElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'none';
            console.log(`Hidden element: ${id}`);
        }
    });

    // Show normal text for PDB upload
    const pdbText = document.getElementById('pdbUploadText');
    const pdbTextDisabled = document.getElementById('pdbUploadTextDisabled');
    if (pdbText) pdbText.style.display = '';
    if (pdbTextDisabled) pdbTextDisabled.style.display = 'none';

    // Restore fetch help text and input placeholder
    const fetchHelpText = document.getElementById('fetchHelpText');
    const fetchHelpTextDisabled = document.getElementById('fetchHelpTextDisabled');
    const pdbIdInput = document.getElementById('pdbOrIdInput');
    const fetchBtn = document.getElementById('fetchStructureBtn');

    if (fetchHelpText) fetchHelpText.style.display = '';
    if (fetchHelpTextDisabled) fetchHelpTextDisabled.style.display = 'none';

    // Re-enable fetch and input
    if (fetchBtn) fetchBtn.style.display = '';
    if (pdbIdInput) {
        pdbIdInput.placeholder = 'Enter PDB ID (e.g., 1ABC) or UniProt name (e.g., P42212)';
        pdbIdInput.readOnly = false;
    }
}

/**
 * Update UI based on ColabFold configuration
 */
export function updateColabFoldUI() {
    const config = window.appConfig || { colabfold_enabled: false };

    if (!config.colabfold_enabled) {
        console.log('ColabFold is disabled - hiding related UI elements');
        hideColabFoldElements();

        // Update upload section text to indicate structure requirement
        const uploadHeader = document.querySelector('h4[id="uploadSection"]');
        if (uploadHeader) {
            uploadHeader.innerHTML = '<i class="fas fa-upload me-2"></i>Upload Structure with pLDDT Scores';
        }

        // Add warning about structure requirement
        const uploadSection = document.querySelector('#structuresCard .card-body');
        if (uploadSection) {
            const existingWarning = document.getElementById('colabfoldDisabledWarning');
            if (!existingWarning) {
                const warning = document.createElement('div');
                warning.id = 'colabfoldDisabledWarning';
                warning.className = 'alert alert-warning mt-3';
                warning.innerHTML = `
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>ColabFold is disabled</strong>. Please upload structure files (PDB/mmCIF) that contain pLDDT scores.
                    FASTA-only uploads are not supported in this mode.
                `;
                uploadSection.appendChild(warning);
            }
        }
    } else {
        console.log('ColabFold is enabled - showing related UI elements');
        showColabFoldElements();

        // Remove warning if it exists
        const warning = document.getElementById('colabfoldDisabledWarning');
        if (warning) {
            warning.remove();
        }

        // Restore original upload header
        const uploadHeader = document.querySelector('h4[id="uploadSection"]');
        if (uploadHeader) {
            uploadHeader.innerHTML = '<i class="fas fa-upload me-2"></i>Upload Structures';
        }
    }
}

/**
 * Initialize ColabFold UI state
 */
export function initializeColabFoldUI() {
    // Wait a bit for DOM to be ready
    setTimeout(() => {
        updateColabFoldUI();
    }, 100);
}
