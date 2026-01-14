// persistence.js - LocalStorage persistence for form state
import { state } from './state.js';

const STORAGE_KEY = 'fpsimp_form_state';
const STORAGE_VERSION = 1;

export function saveState() {
    try {
        const persistentState = {
            version: STORAGE_VERSION,
            timestamp: new Date().toISOString(),
            structures: state.structures,
            membraneRegions: state.membraneRegions,
            fpSites: state.fpSites,
            selectedStructureIndex: state.selectedStructureIndex,
            currentJobId: state.currentJobId,
            lastJobResultsUrl: state.lastJobResultsUrl,
            // Form parameters
            formParams: {
                numFrames: document.getElementById('numFrames')?.value,
                stepsPerFrame: document.getElementById('stepsPerFrame')?.value,
                membrane: document.getElementById('membrane')?.checked,
                membraneWeight: document.getElementById('membraneWeight')?.value,
                barrierRadius: document.getElementById('barrierRadius')?.value,
                kCenter: document.getElementById('kCenter')?.value,
                centerInitial: document.getElementById('centerInitial')?.checked,
                runColabfold: document.getElementById('runColabfold')?.checked,
                useStructurePlddt: document.getElementById('useStructurePlddt')?.checked,
                gpuRelax: document.getElementById('gpuRelax')?.checked,
                // ColabFold parameters
                modelType: document.getElementById('modelType')?.value,
                colabfoldArgs: document.getElementById('colabfoldArgs')?.value,
                // Segmentation
                plddtThreshold: document.getElementById('plddtThreshold')?.value,
                minRigidLength: document.getElementById('minRigidLength')?.value,
                beadSize: document.getElementById('beadSize')?.value,
                membraneSeqInput: document.getElementById('membraneSeqInput')?.value
            }
        };

        localStorage.setItem(STORAGE_KEY, JSON.stringify(persistentState));
        console.log('State saved to localStorage');
    } catch (error) {
        console.error('Error saving state:', error);
    }
}

export function loadState() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (!stored) {
            console.log('No saved state found');
            return false;
        }

        const persistentState = JSON.parse(stored);

        // Check version compatibility
        if (persistentState.version !== STORAGE_VERSION) {
            console.warn('Saved state version mismatch, clearing');
            clearState();
            return false;
        }

        // Restore structures
        if (persistentState.structures && persistentState.structures.length > 0) {
            state.structures = persistentState.structures;
            state.selectedStructureIndex = persistentState.selectedStructureIndex || 0;

            // Update upload context from first structure
            if (state.structures.length > 0) {
                const firstStruct = state.structures[0];
                state.uploadId = firstStruct.upload_id;
                state.isPdbUpload = firstStruct.type === 'pdb';
                state.pdbPath = firstStruct.file_path;
                if (firstStruct.sequences && firstStruct.sequences.length > 0) {
                    state.uploadedSequences = firstStruct.sequences;
                    state.currentSequence = firstStruct.sequences[0];
                }
            }
        }

        // Restore membrane regions
        if (persistentState.membraneRegions) {
            state.membraneRegions = persistentState.membraneRegions;
        }

        // Restore FP sites
        if (persistentState.fpSites) {
            state.fpSites = persistentState.fpSites;
        }

        // Restore current job ID
        if (persistentState.currentJobId) {
            state.currentJobId = persistentState.currentJobId;
        }

        // Restore last job results URL
        if (persistentState.lastJobResultsUrl) {
            state.lastJobResultsUrl = persistentState.lastJobResultsUrl;
        }

        // Restore form parameters
        if (persistentState.formParams) {
            const params = persistentState.formParams;

            // Restore input values
            if (params.numFrames !== undefined) {
                const el = document.getElementById('numFrames');
                if (el) el.value = params.numFrames;
            }
            if (params.stepsPerFrame !== undefined) {
                const el = document.getElementById('stepsPerFrame');
                if (el) el.value = params.stepsPerFrame;
            }
            if (params.membraneWeight !== undefined) {
                const el = document.getElementById('membraneWeight');
                if (el) el.value = params.membraneWeight;
            }
            if (params.barrierRadius !== undefined) {
                const el = document.getElementById('barrierRadius');
                if (el) el.value = params.barrierRadius;
            }
            if (params.kCenter !== undefined) {
                const el = document.getElementById('kCenter');
                if (el) el.value = params.kCenter;
            }
            if (params.plddtThreshold !== undefined) {
                const el = document.getElementById('plddtThreshold');
                if (el) el.value = params.plddtThreshold;
            }
            if (params.minRigidLength !== undefined) {
                const el = document.getElementById('minRigidLength');
                if (el) el.value = params.minRigidLength;
            }
            if (params.beadSize !== undefined) {
                const el = document.getElementById('beadSize');
                if (el) el.value = params.beadSize;
            }
            if (params.modelType !== undefined) {
                const el = document.getElementById('modelType');
                if (el) el.value = params.modelType;
            }
            if (params.colabfoldArgs !== undefined) {
                const el = document.getElementById('colabfoldArgs');
                if (el) el.value = params.colabfoldArgs;
            }
            if (params.membraneSeqInput !== undefined) {
                const el = document.getElementById('membraneSeqInput');
                if (el) el.value = params.membraneSeqInput;
            }

            // Restore checkboxes
            if (params.membrane !== undefined) {
                const el = document.getElementById('membrane');
                if (el) el.checked = params.membrane;
            }
            if (params.centerInitial !== undefined) {
                const el = document.getElementById('centerInitial');
                if (el) el.checked = params.centerInitial;
            }
            if (params.runColabfold !== undefined) {
                const el = document.getElementById('runColabfold');
                if (el) el.checked = params.runColabfold;
            }
            if (params.useStructurePlddt !== undefined) {
                const el = document.getElementById('useStructurePlddt');
                if (el) el.checked = params.useStructurePlddt;
            }
            if (params.gpuRelax !== undefined) {
                const el = document.getElementById('gpuRelax');
                if (el) el.checked = params.gpuRelax;
            }
        }

        console.log('State restored from localStorage:', persistentState.timestamp);
        return true;
    } catch (error) {
        console.error('Error loading state:', error);
        return false;
    }
}

export function clearState() {
    try {
        localStorage.removeItem(STORAGE_KEY);
        console.log('State cleared from localStorage');
    } catch (error) {
        console.error('Error clearing state:', error);
    }
}

// Auto-save on form changes
export function setupAutoSave() {
    // Debounce function to avoid excessive saves
    let saveTimeout = null;
    const debouncedSave = () => {
        if (saveTimeout) clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => {
            saveState();
        }, 500); // Save 500ms after last change
    };

    // Listen to all form inputs
    const formInputs = document.querySelectorAll('input, select, textarea');
    formInputs.forEach(input => {
        input.addEventListener('change', debouncedSave);
        input.addEventListener('input', debouncedSave);
    });

    console.log('Auto-save enabled for', formInputs.length, 'form elements');
}

// Expose functions globally for manual saving
window.fpsimp_saveState = saveState;
window.fpsimp_loadState = loadState;
window.fpsimp_clearState = clearState;
