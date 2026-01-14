import { initializeEventListeners } from './events.js';
import { initializeViewer } from './viewer.js';
import { renderSelectedStructures, populateSequenceSelectorFromAllStructures, updateRegionCount } from './ui.js';
import { loadState, setupAutoSave, saveState } from './persistence.js';
import { startStatusPolling } from './monitoring.js';
import { initializeColabFoldUI } from './colabfold-ui.js';
import { handleSequenceSelect } from './sequence.js';
import { highlightFromTextField, updateSelectorHighlightFromTextField } from './membrane.js';

// Suppress Chrome extension errors that don't affect our app
window.addEventListener('error', (event) => {
    if (event.message && (
        event.message.includes('Could not establish connection') ||
        event.message.includes('Receiving end does not exist')
    )) {
        event.preventDefault();
        return false;
    }
});

// Also suppress unhandled promise rejections from extensions
window.addEventListener('unhandledrejection', (event) => {
    if (event.reason && event.reason.message && (
        event.reason.message.includes('Could not establish connection') ||
        event.reason.message.includes('Receiving end does not exist')
    )) {
        event.preventDefault();
        return false;
    }
});

document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== Application Starting ===');

    // Fetch application configuration
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            const config = await response.json();
            // Store in global scope for other modules to access
            window.appConfig = config;
            console.log('App config loaded:', config);
        }
    } catch (error) {
        console.warn('Failed to fetch app config:', error);
        // Default to ColabFold disabled if config fetch fails
        window.appConfig = { colabfold_enabled: false, disable_colabfold: true };
    }

    // Load saved state from localStorage
    const stateRestored = loadState();
    if (stateRestored) {
        console.log('Previous session restored');
        // Re-render UI with restored state
        renderSelectedStructures();
        populateSequenceSelectorFromAllStructures();

        // Update membrane UI
        handleSequenceSelect();
        updateSelectorHighlightFromTextField();

        // Populate FP chain selectors from restored state
        import('./fp.js').then(module => {
            module.populateFPChainSelectors();
        }).catch(err => console.error('Error loading fp module:', err));

        // Restore job status if there was an active job
        await restoreJobStatus();
    }

    // Initialize event listeners
    initializeEventListeners();

    // Initialize structure list display
    renderSelectedStructures();

    // Setup auto-save for form changes
    setupAutoSave();

    // Start ColabFold status monitoring
    startStatusPolling();

    // Initialize ColabFold UI based on configuration
    initializeColabFoldUI();

    // Initialize 3D viewer (non-blocking)
    try {
        const available = await initializeViewer();
        if (available) {
            console.log('3D viewer initialized successfully');

            // Reconstruct 3D viewer if state was restored and we have a structure
            if (stateRestored) {
                await reconstructViewer();
            }
        } else {
            console.log('3D viewer not available - continuing without 3D preview');
        }
    } catch (error) {
        console.warn('3D viewer initialization failed:', error);
    }

    console.log('=== Application Initialized ===');
});

// Restore job status if there was an active job
async function restoreJobStatus() {
    const { state } = await import('./state.js');
    const { showJobStatus, startStatusPolling } = await import('./job.js');

    if (state.currentJobId) {
        console.log('Restoring job status for:', state.currentJobId);
        try {
            // Fetch current job status
            const response = await fetch(`/api/job_status/${state.currentJobId}`);
            if (response.ok) {
                const data = await response.json();

                // Save results URL if available
                if (data.results_url) {
                    state.lastJobResultsUrl = data.results_url;
                }

                // Show job status card
                showJobStatus(state.currentJobId, data.results_url);

                // Only start polling if job is still active
                if (data.status === 'queued' || data.status === 'running' ||
                    data.status === 'colabfold_complete' || data.status === 'sampling_complete') {
                    startStatusPolling();
                    console.log('Job status restored - polling started');
                } else {
                    // Job is done, clear currentJobId but keep results URL
                    state.currentJobId = null;

                    // Save state so next refresh shows results-only mode
                    const { saveState } = await import('./persistence.js');
                    saveState();

                    console.log('Job completed, showing results');
                }
            } else {
                // Job not found, clear it
                state.currentJobId = null;
                console.log('Job not found, clearing from state');
            }
        } catch (error) {
            console.warn('Failed to restore job status:', error);
            state.currentJobId = null;
        }
    } else if (state.lastJobResultsUrl) {
        // No active job but we have results from a previous job
        console.log('Showing results from previous job');
        showJobStatus(null, state.lastJobResultsUrl);
    }
}

// Reconstruct 3D viewer from saved state
async function reconstructViewer() {
    const { state } = await import('./state.js');
    const { showStructurePreview } = await import('./viewer.js');

    // Find the selected structure or the first PDB structure
    let structureToShow = null;

    if (state.selectedStructureIndex >= 0 && state.structures[state.selectedStructureIndex]) {
        structureToShow = state.structures[state.selectedStructureIndex];
    } else {
        // Find first PDB structure with a file_url
        structureToShow = state.structures.find(s => s.type === 'pdb' && s.file_url);
    }

    if (structureToShow && structureToShow.file_url) {
        try {
            console.log('Reconstructing 3D viewer for:', structureToShow.filename);
            await showStructurePreview(
                structureToShow.file_url,
                structureToShow.filename,
                structureToShow.sequences
            );
        } catch (error) {
            console.warn('Failed to reconstruct 3D viewer:', error);
        }
    }
}

// Save state before page unload
window.addEventListener('beforeunload', () => {
    saveState();
});
