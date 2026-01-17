import { DOM } from './config.js';
import { handleFileUpload, handlePdbUpload, handleFetchStructure, clearSelectedStructures, recalculateSegmentation } from './api.js';
import { handleSequenceSelect } from './sequence.js';
import { handleMembraneSeqInputEvent, clearMembraneRegions, highlightFromTextField, updateSelectorHighlightFromTextField } from './membrane.js';
import { updateRegionCount } from './ui.js';
import { submitJob } from './job.js';
import { toggleFPSiteSelection, setupFPSiteListeners } from './fp.js';

export function initializeEventListeners() {
    const fastaFileInput = document.getElementById(DOM.fastaFile);
    if (fastaFileInput) {
        fastaFileInput.addEventListener('change', (e) => handleFileUpload(e));
    }

    const browsePdbBtn = document.getElementById('browsePdbBtn');
    if (browsePdbBtn) {
        browsePdbBtn.addEventListener('click', () => {
            const fileInput = document.getElementById(DOM.pdbFile);
            if (fileInput) fileInput.click();
        });
    }

    const clearPdbBtn = document.getElementById(DOM.clearPdbBtn);
    if (clearPdbBtn) {
        clearPdbBtn.addEventListener('click', () => {
            const input = document.getElementById(DOM.pdbIdInput);
            const fileInput = document.getElementById(DOM.pdbFile);
            if (input) input.value = '';
            if (fileInput) fileInput.value = '';

            // Also clear all uploaded/selected structures
            clearSelectedStructures();
        });
    }


    const pdbFileInput = document.getElementById(DOM.pdbFile);
    if (pdbFileInput) {
        pdbFileInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                const textInput = document.getElementById(DOM.pdbIdInput);
                if (textInput) textInput.value = e.target.files[0].name;
            }
            await handlePdbUpload(e);

            // Clear text input after upload
            const textInput = document.getElementById(DOM.pdbIdInput);
            if (textInput) textInput.value = '';
        });
    }

    const fetchStructureBtn = document.getElementById(DOM.fetchStructureBtn);
    if (fetchStructureBtn) {
        fetchStructureBtn.addEventListener('click', () => handleFetchStructure());
    }

    const clearBtn = document.getElementById(DOM.clearStructuresBtn);
    if (clearBtn) {
        clearBtn.addEventListener('click', () => clearSelectedStructures());
    }

    const sequenceSelect = document.getElementById(DOM.sequenceSelect);
    if (sequenceSelect) {
        sequenceSelect.addEventListener('change', () => handleSequenceSelect());
    }

    const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);
    if (membraneSeqInput) {
        const onMembraneInput = (e) => {
            handleMembraneSeqInputEvent(e);
            // Explicitly call update functions to ensure UI sync
            updateRegionCount();
            highlightFromTextField();
            updateSelectorHighlightFromTextField();
        };
        membraneSeqInput.addEventListener('input', onMembraneInput);
        membraneSeqInput.addEventListener('change', onMembraneInput);
        membraneSeqInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                handleMembraneSeqInputEvent(e, true);
            }
        });
        setTimeout(() => {
            highlightFromTextField();
        }, 0);
    }

    const clearRegions = document.getElementById(DOM.clearRegions);
    if (clearRegions) {
        clearRegions.addEventListener('click', () => clearMembraneRegions());
    }

    const parametersForm = document.getElementById(DOM.parametersForm);
    if (parametersForm) {
        parametersForm.addEventListener('submit', (e) => submitJob(e));
    }

    const measureCheckbox = document.getElementById(DOM.measure);
    if (measureCheckbox) {
        measureCheckbox.addEventListener('change', (e) => toggleFPSiteSelection(e.target.checked));
    }

    setupFPSiteListeners();
    setupPipelineParametersToggle();
    setupSegmentationListeners();
}

function setupSegmentationListeners() {
    const params = ['plddtThreshold', 'minRigidLength', 'beadSize', 'modelDisorderedAsBeads'];
    params.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', () => {
                recalculateSegmentation();
            });
        }
    });
}
function setupPipelineParametersToggle() {
    const toggleButton = document.querySelector(`[data-bs-target="#${DOM.pipelineParametersCollapse}"]`);
    const collapseElement = document.getElementById(DOM.pipelineParametersCollapse);

    if (toggleButton && collapseElement) {
        collapseElement.addEventListener('show.bs.collapse', () => {
            const icon = toggleButton.querySelector('i');
            icon.className = 'fas fa-chevron-up me-1';
            toggleButton.childNodes[toggleButton.childNodes.length - 1].textContent = 'Hide Parameters';
        });

        collapseElement.addEventListener('hide.bs.collapse', () => {
            const icon = toggleButton.querySelector('i');
            icon.className = 'fas fa-chevron-down me-1';
            toggleButton.childNodes[toggleButton.childNodes.length - 1].textContent = 'Show Parameters';
        });
    }
}
