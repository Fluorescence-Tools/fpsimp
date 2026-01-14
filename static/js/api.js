import { state } from './state.js';
import { showSuccess, showError, setLoadingState, formatFileSize, populateSequenceSelector, populateSequenceSelectorFromAllStructures, renderSelectedStructures, showInfo } from './ui.js';
import { handleSequenceSelect } from './sequence.js';
import { showStructurePreview, hideStructureViewer } from './viewer.js';
import { DOM } from './config.js';
import { saveState } from './persistence.js';

export async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    setLoadingState(true);

    const formData = new FormData();
    formData.append('fasta', file);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (response.ok) {
            // If we have multiple sequences, create separate structure entries for each
            let addedCount = 0;
            if (data.sequences && data.sequences.length > 1) {
                // Multi-sequence FASTA: create separate entries for each sequence
                data.sequences.forEach((seq, index) => {
                    const newStructure = state.addStructure({
                        upload_id: data.upload_id,
                        filename: `${data.filename}_${seq.id}`, // Make filename unique for each sequence
                        file_path: null,
                        file_url: null,
                        file_size: Math.ceil(data.file_size / data.sequences.length), // Approximate size per sequence
                        sequences: [seq], // Only this sequence for this structure
                        type: 'fasta',
                        selected: false, // Default to unselected for multi-seq FASTA
                        original_filename: data.filename, // Keep track of original file
                        sequence_index: index
                    });
                    addedCount++;
                });
            } else if (data.sequences && data.sequences.length === 1) {
                // Single sequence FASTA: treat as before (default selected)
                const newStructure = state.addStructure({
                    upload_id: data.upload_id,
                    filename: data.filename,
                    file_path: null,
                    file_url: null,
                    file_size: data.file_size,
                    sequences: data.sequences || [],
                    type: 'fasta',
                    selected: true, // Default to selected for single sequence
                });
                addedCount = 1;
            }

            // Render structures list and sequence options from selected items
            renderSelectedStructures();
            saveState();

            // Update sequence selector with all sequences from all structures
            const allSequences = populateSequenceSelectorFromAllStructures();
            if (allSequences.length > 0) {
                const sequenceSelect = document.getElementById('sequenceSelect');
                if (sequenceSelect) {
                    // Select the first sequence from the newly added structures
                    const newSequences = allSequences.filter(s =>
                        s.structureName.startsWith(data.filename) || s.structureName === data.filename
                    );
                    if (newSequences.length > 0) {
                        sequenceSelect.value = newSequences[0].id;
                    }
                    handleSequenceSelect();
                }
            }

            // ColabFold checkbox remains user-controlled (default checked for pLDDT scores)

            showSuccess(`Uploaded ${data.filename} (${formatFileSize(data.file_size)}) - ${addedCount} sequence${addedCount === 1 ? '' : 's'} added (uncheck to exclude from multimer)`);
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('Upload failed: ' + error.message);
    } finally {
        setLoadingState(false);
    }
}

export async function handleFetchStructure() {
    const input = document.getElementById(DOM.pdbIdInput);
    if (!input) return;

    const id = (input.value || '').trim();
    if (!id) {
        showError('Please enter a PDB ID (e.g., 1ABC), UniProt name (e.g., P42212), or AlphaFold ID (e.g., AF-P32455-F1-model_v6)');
        return;
    }

    // Check if ColabFold is disabled and validate ID type
    const config = window.appConfig || { colabfold_enabled: false };
    if (!config.colabfold_enabled) {
        // When ColabFold is disabled, only allow UniProt names
        if (!/^[A-Z0-9]{6,}$/.test(id)) {
            showError('ColabFold is disabled. Only UniProt names (e.g., P42212) are allowed.');
            return;
        }
    }

    await fetchStructureById(id, input);
}

async function fetchStructureById(id, inputElement) {

    setLoadingState(true);

    try {
        const response = await fetch('/api/fetch_structure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });

        const data = await response.json();

        if (response.ok) {
            // Add the PDB structure to the state
            const newStructure = state.addStructure({
                upload_id: data.upload_id,
                filename: data.filename,
                file_path: data.file_path,
                file_url: data.file_url || null,
                file_size: data.file_size,
                sequences: data.sequences || [],
                derived_fasta: data.derived_fasta || null,
                uniprot_id: data.uniprot_id || null, // Store UniProt ID if available
                selected: true,  // Default to selected
                type: 'pdb'  // Keep as PDB for display purposes
            });

            // Update the UI
            renderSelectedStructures();
            saveState();

            // Show 3D preview if available
            if (data.file_url) {
                try {
                    await showStructurePreview(data.file_url, data.filename, data.sequences);
                } catch (viewerError) {
                    console.warn('3D preview failed:', viewerError);
                }
            }

            // Update sequence selector with all sequences (including from PDB)
            const allSequences = populateSequenceSelectorFromAllStructures();
            if (allSequences.length > 0) {
                const sequenceSelect = document.getElementById('sequenceSelect');
                if (sequenceSelect) {
                    // Select the first sequence from the newly added structure
                    const newStructureSequences = allSequences.filter(seq =>
                        seq.structureName === data.filename
                    );
                    if (newStructureSequences.length > 0) {
                        sequenceSelect.value = newStructureSequences[0].id;
                    }
                    handleSequenceSelect();
                }
            }

            // ColabFold checkbox remains user-controlled (default checked for pLDDT scores)

            // Clear the input
            inputElement.value = '';

            const sequenceInfo = data.sequences?.length > 0
                ? `with ${data.sequences.length} chain(s)`
                : (data.uniprot_id ? 'with UniProt sequence' : '');

            showSuccess(`Fetched ${data.filename} (${formatFileSize(data.file_size)}) ${sequenceInfo} - ready for prediction`);
        } else {
            showError(data.error || 'Failed to fetch structure');
        }
    } catch (error) {
        console.error('Fetch error:', error);
        showError('Fetch failed: ' + (error.message || 'Unknown error'));
    } finally {
        setLoadingState(false);
    }
}

// Handle PDB file upload
export async function handlePdbUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    setLoadingState(true);

    try {
        const formData = new FormData();
        formData.append('pdb', file);

        console.log('Uploading PDB file:', file.name, 'Size:', file.size, 'Type:', file.type);
        console.log('FormData keys:', Array.from(formData.keys()));

        const response = await fetch('/api/upload_pdb', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Add the PDB structure to the state
            const newStructure = state.addStructure({
                upload_id: data.upload_id,
                filename: data.filename,
                file_path: data.file_path,
                file_url: data.file_url || null,
                file_size: data.file_size,
                sequences: data.sequences || [],
                derived_fasta: data.derived_fasta || null,
                selected: true,  // Default to selected
                type: 'pdb'  // Keep as PDB for display purposes
            });

            // Update the UI
            renderSelectedStructures();

            // Show 3D preview if available
            if (data.file_url) {
                try {
                    await showStructurePreview(data.file_url, file.name, data.sequences);
                } catch (e) {
                    console.error('Failed to show structure preview:', e);
                }
            }

            // Update sequence selector with all sequences from all structures (including PDB sequences)
            const allSequences = populateSequenceSelectorFromAllStructures();
            if (allSequences.length > 0) {
                const sequenceSelect = document.getElementById('sequenceSelect');
                if (sequenceSelect) {
                    // Select the first sequence from the newly added structure
                    const newStructureSequences = allSequences.filter(seq =>
                        seq.structureName === data.filename
                    );
                    if (newStructureSequences.length > 0) {
                        sequenceSelect.value = newStructureSequences[0].id;
                    }
                    handleSequenceSelect();
                }
            }

            // ColabFold checkbox remains user-controlled (default checked for pLDDT scores)

            showSuccess(`Uploaded ${data.filename} (${formatFileSize(data.file_size)}) with ${data.sequences?.length || 0} chain(s) - sequences available for prediction`);
        } else {
            showError(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showError('Upload failed: ' + (error.message || 'Unknown error'));
    } finally {
        setLoadingState(false);
        // Reset file input
        event.target.value = '';
    }
}

// Clear all selected structures and all stored settings
export async function clearSelectedStructures() {
    // Clear all application state
    state.clearStructures();
    renderSelectedStructures();

    // Clear all localStorage data
    const { clearState } = await import('./persistence.js');
    clearState();
    
    // Clear all sessionStorage data
    for (let i = 0; i < sessionStorage.length; i++) {
        const key = sessionStorage.key(i);
        if (key && key.startsWith('lastReload_')) {
            sessionStorage.removeItem(key);
        }
    }

    // Reset all form fields to default values
    resetAllFormFields();

    // Hide the viewer
    hideStructureViewer();

    // Clear any existing sequences
    state.uploadedSequences = [];
    state.derivedFasta = null;
    state.pdbPath = null;
    state.uploadId = null;
    state.currentJobId = null;
    state.lastJobResultsUrl = null;
    state.membraneRegions = {};
    state.fpSites = {
        donor: { site1: null, site2: null },
        acceptor: { site1: null, site2: null }
    };

    console.log('All data and settings cleared');
}

// Reset all form fields to their default values
function resetAllFormFields() {
    // Reset numeric inputs
    const numericInputs = [
        'numFrames', 'stepsPerFrame', 'membraneWeight', 'barrierRadius', 
        'kCenter', 'plddtThreshold', 'minRigidLength', 'beadSize'
    ];
    numericInputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.value = element.defaultValue || '';
        }
    });

    // Reset FP position inputs (residue indices)
    const fpPositionInputs = [
        'donorPos1', 'donorPos2', 'acceptorPos1', 'acceptorPos2'
    ];
    fpPositionInputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.value = '';
            element.classList.remove('is-valid', 'is-invalid');
            element.setCustomValidity('');
            // Reset placeholder and max attributes
            element.removeAttribute('max');
            element.placeholder = 'AA Position';
            element.title = 'Select a chain first, then enter residue index';
        }
    });

    // Reset select elements
    const selectElements = ['modelType'];
    selectElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.selectedIndex = 0;
        }
    });

    // Reset FP chain selectors
    const fpChainSelectors = [
        'donorChain1', 'donorChain2', 'acceptorChain1', 'acceptorChain2'
    ];
    fpChainSelectors.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.value = '';
        }
    });

    // Reset checkboxes
    const checkboxes = [
        'membrane', 'centerInitial', 'runColabfold', 'useStructurePlddt', 
        'gpuRelax', 'measure'
    ];
    checkboxes.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.checked = element.defaultChecked || false;
        }
    });

    // Reset textareas
    const textareas = ['colabfoldArgs', 'membraneSeqInput'];
    textareas.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.value = '';
        }
    });

    // Update the sequence selector
    const sequenceSelect = document.getElementById('sequenceSelect');
    if (sequenceSelect) {
        sequenceSelect.innerHTML = '<option value="">No sequence available</option>';
    }

    showInfo('Cleared all structures and settings');
}

// Show structure by index
export async function showStructureByIndex(index) {
    const structure = state.selectStructure(index);
    if (structure && structure.file_url) {
        try {
            await showStructurePreview(structure.file_url, structure.filename, structure.sequences);
            renderSelectedStructures(); // Update UI to show selection
            return true;
        } catch (error) {
            console.error('Failed to show structure preview:', error);
            return false;
        }
    }
    return false;
}

// Remove structure by index
export function removeStructure(index) {
    if (confirm('Are you sure you want to remove this structure?')) {
        state.removeStructure(index);
        renderSelectedStructures();
        saveState();
        // Refresh sequences from remaining selected structures
        populateSequenceSelectorFromAllStructures();

        // Hide viewer if no structures remain
        if (state.structures.length === 0) {
            hideStructureViewer();
        }

        showInfo('Structure removed');
    }
}

// Toggle checkbox selection state
export function toggleStructureSelected(index) {
    if (index >= 0 && index < state.structures.length) {
        state.structures[index].selected = !state.structures[index].selected;
        // Re-render list and refresh sequences
        renderSelectedStructures();
        populateSequenceSelectorFromAllStructures();
        saveState();
    }
}

// Global functions for HTML onclick handlers
window.showStructureByIndex = showStructureByIndex;
window.removeStructure = removeStructure;
window.toggleStructureSelected = toggleStructureSelected;
