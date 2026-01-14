import { DOM } from './config.js';
import { state } from './state.js';

export function showAlert(message, { type = 'info', html = false } = {}) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'btn-close';
    closeBtn.setAttribute('data-bs-dismiss', 'alert');

    if (html) {
        const messageDiv = document.createElement('div');
        messageDiv.innerHTML = message;
        alert.appendChild(messageDiv);
    } else {
        alert.textContent = message;
    }

    alert.appendChild(closeBtn);

    const container = document.querySelector(DOM.alertContainer);
    if (container.firstChild) {
        container.insertBefore(alert, container.firstChild);
    } else {
        container.appendChild(alert);
    }

    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

export function showSuccess(message, options = {}) {
    showAlert(message, { ...options, type: 'success' });
}

export function showError(message, options = {}) {
    showAlert(message, { ...options, type: 'danger' });
}

export function showInfo(message, options = {}) {
    showAlert(message, { ...options, type: 'info' });
}

export function setLoadingState(loading) {
    const submitBtn = document.querySelector(DOM.submitBtn);
    if (!submitBtn) {
        console.warn('Submit button not found');
        return;
    }
    if (loading) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    } else {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-play me-2"></i>Submit Job';
    }
}

export function formatFileSize(bytes) {
    // Handle undefined, null, or non-numeric values
    if (bytes == null || isNaN(bytes) || bytes < 0) {
        return 'Unknown size';
    }
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function populateSequenceSelector(sequences) {
    console.log('populateSequenceSelector called with', sequences.length, 'sequences:', sequences.map(s => s.id));

    const sequenceSelector = document.getElementById(DOM.sequenceSelector);
    const sequenceSelect = document.getElementById(DOM.sequenceSelect);
    const sequenceDisplay = document.getElementById(DOM.sequenceDisplay);

    const paramsCard = document.getElementById(DOM.parametersCard);
    if (paramsCard) {
        paramsCard.style.display = 'block';
    }
    if (sequenceDisplay) {
        sequenceDisplay.style.display = 'block';
    }

    if (sequenceSelector && sequenceSelect) {
        sequenceSelector.style.display = 'block';
        sequenceSelect.innerHTML = '';

        sequences.forEach((seq, idx) => {
            console.log(`Adding option ${idx}: ${seq.id} (${seq.sequence.length} aa)`);
            const option = document.createElement('option');
            option.value = seq.id;
            option.textContent = `${seq.id} (${seq.sequence.length} aa)`;
            sequenceSelect.appendChild(option);
            if (idx === 0) {
                sequenceSelect.value = seq.id;
            }
        });

        console.log('Final dropdown has', sequenceSelect.options.length, 'options');
    }
}

export function populateSequenceSelectorFromAllStructures() {
    console.log('populateSequenceSelectorFromAllStructures called, structures:', state.structures?.length || 0);
    const selectedStructures = (state.structures || []).filter(s => s.selected && s.sequences && s.sequences.length > 0);

    // Collect sequences from selected structures (FASTA and PDB)
    const allSequences = [];
    selectedStructures.forEach((structure) => {
        if (structure.sequences && structure.sequences.length > 0) {
            structure.sequences.forEach(seq => {
                allSequences.push({
                    ...seq,
                    id: `${structure.filename}_${seq.id}`, // Keep prefixed ID to avoid conflicts
                    structureIndex: structure.id,
                    structureName: structure.filename,
                    originalId: seq.id,
                    structureType: structure.type, // Track whether it's from FASTA or PDB
                    selected: true
                });
            });
        }
    });

    const selectorWrap = document.getElementById(DOM.sequenceSelector);
    const sequenceSelect = document.getElementById(DOM.sequenceSelect);

    console.log('DOM elements found:', {
        selectorWrap: !!selectorWrap,
        sequenceSelect: !!sequenceSelect,
        allSequences: allSequences.length,
        selectedStructures: selectedStructures.length
    });

    if (selectorWrap && sequenceSelect) {
        selectorWrap.style.display = 'block';
        sequenceSelect.innerHTML = '<option value="">Select a sequence for membrane anchor selection...</option>';

        allSequences.forEach((seq, idx) => {
            console.log(`Adding option ${idx}: ${seq.originalId} (${seq.sequence.length} aa) from ${seq.structureName} (${seq.structureType})`, seq.fp_name ? `[FP: ${seq.fp_name}]` : '');
            const option = document.createElement('option');
            option.value = seq.id; // Keep the prefixed ID for internal use

            // Build option text with FP indicator if detected
            let optionText = `${seq.originalId} (${seq.sequence.length} aa)`;
            if (seq.fp_name) {
                optionText += ` 🔬 ${seq.fp_name}`;
            }
            optionText += ` from ${seq.structureName}`;

            option.textContent = optionText;
            sequenceSelect.appendChild(option);
            if (idx === 0) {
                sequenceSelect.value = seq.id;
            }
        });

        console.log('Sequence selector populated with', allSequences.length, 'sequences from structures');
    } else {
        console.warn('Sequence selector elements not found:', {
            selectorWrap: !!selectorWrap,
            sequenceSelect: !!sequenceSelect
        });
    }

    // Update multimer sequence display whenever selection changes
    updateMultimerSequenceDisplay();

    // Update global state for FP measurements
    state.uploadedSequences = allSequences;

    // Update FP measurement chain selectors
    import('./fp.js').then(module => {
        module.populateFPChainSelectors();
        // Fully auto-populate if possible
        module.fullyAutoPopulateFPs();
    }).catch(err => console.error('Error loading fp module:', err));

    return allSequences;
}


export function updateMultimerSequenceDisplay() {
    console.log('Updating multimer sequence display');

    const selectedStructures = state.getSelectedStructures();
    const multimerDisplay = document.getElementById(DOM.multimerSequenceDisplay);
    const multimerText = document.getElementById(DOM.multimerSequenceText);
    const sequenceDisplay = document.getElementById(DOM.sequenceDisplay);
    const sequenceText = document.getElementById(DOM.sequenceText);

    console.log('DOM elements found:', {
        multimerDisplay: !!multimerDisplay,
        multimerText: !!multimerText,
        sequenceDisplay: !!sequenceDisplay,
        sequenceText: !!sequenceText,
        selectedStructures: selectedStructures.length
    });

    if (!multimerDisplay || !multimerText) {
        console.warn('Multimer sequence display elements not found');
        return;
    }

    // Only show for structures that have sequences (FASTA and PDB structures)
    const structuresWithSequences = selectedStructures.filter(s => s.sequences && s.sequences.length > 0);

    if (structuresWithSequences.length === 0) {
        multimerDisplay.style.display = 'none';
        if (sequenceDisplay) sequenceDisplay.style.display = 'none';

        // Clear membraneSeqInput when no structures with sequences
        updateMembraneSeqInput([]);
        return;
    }

    // Collect all sequences from selected structures (FASTA and PDB)
    const allSequences = [];
    structuresWithSequences.forEach(structure => {
        if (structure.sequences && structure.sequences.length > 0) {
            structure.sequences.forEach(seq => {
                allSequences.push({
                    structureName: structure.filename,
                    sequenceId: seq.id,
                    sequence: seq.sequence,
                    structureType: structure.type,
                    // Include all FP detection fields
                    fp_name: seq.fp_name,
                    fp_color: seq.fp_color,
                    fp_match_type: seq.fp_match_type,
                    fp_motif: seq.fp_motif,
                    fp_start: seq.fp_start,
                    fp_end: seq.fp_end,
                    fps: seq.fps
                });
            });
        }
    });

    if (allSequences.length === 0) {
        multimerDisplay.style.display = 'none';
        if (sequenceDisplay) sequenceDisplay.style.display = 'none';
        return;
    }

    // Create the multimer sequence (sequences separated by ':')
    const multimerSequence = allSequences.map(seq => seq.sequence).join(':');

    // Create a formatted display with structure/sequence info
    const sequenceInfo = allSequences.map((seq, index) =>
        `# Chain ${String.fromCharCode(65 + index)}: ${seq.sequenceId} from ${seq.structureName} (${seq.sequence.length} aa)`
    ).join('\n');

    const displayText = `${sequenceInfo}\n\n# Multimer sequence for ColabFold:\n${multimerSequence}`;

    multimerText.value = displayText;

    multimerDisplay.style.display = 'block';

    // Show the membrane sequence selector with the multimer sequence
    if (sequenceDisplay && sequenceText) {
        console.log('Showing sequence display for multimer');
        sequenceDisplay.style.display = 'block';

        // Create an interactive display for the multimer sequence
        sequenceText.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.className = 'sequence-wrapper';

        // Add header information
        const header = document.createElement('div');
        header.className = 'alert alert-info py-2 mb-3';
        header.innerHTML = `
            <strong>Multimer Assembly:</strong> ${allSequences.length} chains
            <br><small class="text-muted">Click and drag to select membrane regions across the entire multimer sequence</small>
        `;
        wrapper.appendChild(header);

        try {
            // Create the interactive multimer sequence display
            let currentPosition = 0;
            allSequences.forEach((seq, index) => {
                console.log(`Multimer display - Chain ${index}:`, {
                    id: seq.id,
                    fp_name: seq.fp_name,
                    fp_color: seq.fp_color,
                    fp_match_type: seq.fp_match_type,
                    fp_motif: seq.fp_motif,
                    seqLength: seq.sequence?.length
                });

                const line = document.createElement('div');
                line.className = 'sequence-line';

                const pos = document.createElement('span');
                pos.className = 'position-indicator text-muted';
                pos.textContent = `Chain ${String.fromCharCode(65 + index)}:`.padEnd(8, ' ');
                line.appendChild(pos);

                // Detect FP regions for this sequence (may be multiple)
                let fpRegions = [];

                if (seq.fp_name) {
                    // Check if we have multiple FPs
                    if (seq.fps && seq.fps.length > 0) {
                        // Multiple FPs detected
                        fpRegions = seq.fps.map(fp => ({
                            start: fp.start,
                            end: fp.end,
                            color: fp.color
                        }));
                        console.log(`Multiple FPs in chain: ${seq.fps.map(fp => fp.name).join(', ')}`);
                    } else {
                        // Single FP - use legacy fields
                        if (seq.fp_start !== undefined && seq.fp_end !== undefined) {
                            fpRegions.push({
                                start: seq.fp_start,
                                end: seq.fp_end,
                                color: seq.fp_color || 'green'
                            });
                        } else if (seq.fp_match_type === 'full') {
                            fpRegions.push({
                                start: 0,
                                end: seq.sequence.length - 1,
                                color: seq.fp_color || 'green'
                            });
                        } else if (seq.fp_match_type === 'motif' && seq.fp_motif) {
                            const seqUpper = seq.sequence.toUpperCase();
                            const motifPos = seqUpper.indexOf(seq.fp_motif.toUpperCase());
                            if (motifPos >= 0) {
                                fpRegions.push({
                                    start: motifPos,
                                    end: motifPos + seq.fp_motif.length - 1,
                                    color: seq.fp_color || 'green'
                                });
                            }
                        }
                    }
                    console.log(`FP regions for chain:`, fpRegions);
                }

                // Display the full sequence for interaction
                for (let j = 0; j < seq.sequence.length; j++) {
                    const char = document.createElement('span');
                    char.className = 'sequence-char';
                    char.textContent = seq.sequence[j];
                    char.dataset.position = currentPosition.toString();
                    char.dataset.chain = String.fromCharCode(65 + index);

                    // Check if this position is within any FP region
                    for (const fpRegion of fpRegions) {
                        if (j >= fpRegion.start && j <= fpRegion.end) {
                            char.classList.add('fp-region');
                            if (fpRegion.color === 'green') {
                                char.classList.add('fp-green');
                            } else if (fpRegion.color === 'magenta') {
                                char.classList.add('fp-magenta');
                            } else if (fpRegion.color === 'yellow') {
                                char.classList.add('fp-yellow');
                            }
                            break; // Only apply first matching region
                        }
                    }

                    line.appendChild(char);
                    currentPosition++;
                }

                wrapper.appendChild(line);
            });

            // Add a note about the full sequence
            const note = document.createElement('div');
            note.className = 'text-muted mt-2';
            note.innerHTML = `<small>Full multimer sequence length: ${multimerSequence.length} amino acids</small>`;
            wrapper.appendChild(note);

            sequenceText.appendChild(wrapper);
            console.log('Multimer sequence display created successfully');
        } catch (error) {
            console.error('Error creating multimer sequence display:', error);
            // Fallback to simple text display
            sequenceText.innerHTML = `
                <div class="alert alert-warning">
                    <strong>Error displaying multimer sequence:</strong> ${error.message}
                    <br><small>Full sequence length: ${multimerSequence.length}</small>
                </div>
            `;
        }

        // Create a fake sequence object for membrane region tracking
        state.currentSequence = {
            id: 'multimer_sequence',
            sequence: multimerSequence
        };

        // Initialize membrane regions if not exists
        if (!state.membraneRegions['multimer_sequence']) {
            state.membraneRegions['multimer_sequence'] = [];
        }

        // Set up membrane selection functionality
        setTimeout(() => {
            import('./membrane.js').then(module => {
                module.setupSequenceSelection();
                module.highlightFromTextField();
                updateMembraneDisplay();
                updateRegionCount();
            }).catch(error => {
                console.error('Error loading membrane module:', error);
            });
        }, 0);

        // Display multimer information with different background colors for each chain
        displayMultimerInformation(allSequences);
    }

    // Don't automatically populate membraneSeqInput - let users enter their own identifiers

    console.log(`Multimer sequence generated with ${allSequences.length} chains`);
}

export function updateMembraneSeqInput(allSequences) {
    console.log('Updating membraneSeqInput with sequences:', allSequences.length);

    const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);

    if (!membraneSeqInput) {
        console.warn('membraneSeqInput element not found');
        return;
    }

    if (allSequences.length === 0) {
        membraneSeqInput.value = '';
        return;
    }

    // Generate sequence identifiers for the membrane input
    // Format: SEQA, SEQB, SEQC for each chain
    const sequenceIdentifiers = allSequences.map((seq, index) =>
        `SEQ${String.fromCharCode(65 + index)}`
    ).join(', ');

    membraneSeqInput.value = sequenceIdentifiers;
    console.log('membraneSeqInput populated with:', sequenceIdentifiers);
}

export function displayMultimerInformation(allSequences) {
    const multimerInfoContainer = document.getElementById(DOM.multimerInfoContainer);
    const multimerInfoContent = document.getElementById(DOM.multimerInfoContent);

    if (!multimerInfoContainer || !multimerInfoContent) {
        console.warn('Multimer info elements not found:', {
            multimerInfoContainer: !!multimerInfoContainer,
            multimerInfoContent: !!multimerInfoContent
        });
        return;
    }

    if (allSequences.length === 0) {
        multimerInfoContainer.style.display = 'none';
        return;
    }

    // Chain colors for different backgrounds
    const chainColors = [
        'bg-primary bg-opacity-10 border-primary',
        'bg-success bg-opacity-10 border-success',
        'bg-warning bg-opacity-10 border-warning',
        'bg-info bg-opacity-10 border-info',
        'bg-secondary bg-opacity-10 border-secondary',
        'bg-danger bg-opacity-10 border-danger'
    ];

    // Create multimer information display
    const content = allSequences.map((seq, index) => {
        const colorClass = chainColors[index % chainColors.length];
        const chainLetter = String.fromCharCode(65 + index);

        return `
            <div class="card ${colorClass} mb-3">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h6 class="mb-0">
                        <span class="badge bg-primary me-2">Chain ${chainLetter}</span>
                        ${seq.sequenceId} from ${seq.structureName}
                    </h6>
                    <small class="text-muted">${seq.sequence.length} residues</small>
                </div>
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <small class="text-muted">Sequence:</small>
                        <button class="btn btn-sm btn-outline-primary" onclick="navigator.clipboard.writeText('${seq.sequence.replace(/'/g, "\\'")}')">
                            <i class="fas fa-copy me-1"></i>Copy
                        </button>
                    </div>
                    <div class="sequence-display-small" style="font-family: monospace; font-size: 0.85em; line-height: 1.4; background: rgba(0,0,0,0.05); padding: 8px; border-radius: 4px; max-height: 100px; overflow-y: auto;">
                        ${seq.sequence.match(/.{1,60}/g)?.map(line => `<div>${line}</div>`).join('') || seq.sequence}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Add summary information
    const totalLength = allSequences.reduce((sum, seq) => sum + seq.sequence.length, 0);
    const summary = `
        <div class="alert alert-info mb-3">
            <h6 class="mb-2"><i class="fas fa-info-circle me-1"></i>Multimer Assembly Summary</h6>
            <div class="row g-2">
                <div class="col-6">
                    <strong>Chains:</strong> ${allSequences.length}
                </div>
                <div class="col-6">
                    <strong>Total Length:</strong> ${totalLength} aa
                </div>
            </div>
            <div class="mt-2">
                <small class="text-muted">
                    Each chain is highlighted with a different color for easy identification.
                    Click the copy button to copy individual chain sequences.
                </small>
            </div>
        </div>
    `;

    multimerInfoContent.innerHTML = summary + content;
    multimerInfoContainer.style.display = 'block';

    console.log('Multimer information displayed with', allSequences.length, 'chains');
}

export function displayChainInformation(sequenceId, sequences) {
    const selectedSeq = sequences.find(seq => seq.id === sequenceId);
    if (!selectedSeq) return;

    const chainContainer = document.getElementById(DOM.chainTextFields);
    const chainDisplayContainer = document.getElementById(DOM.chainDisplayContainer);

    if (!chainContainer || !chainDisplayContainer) return;

    const chains = selectedSeq.sequence.split(':');

    if (chains.length > 1) {
        chainContainer.innerHTML = chains.map((chain, index) => `
            <div class="mb-2">
                <label class="form-label">Chain ${String.fromCharCode(65 + index)} (${chain.length} residues):</label>
                <textarea class="form-control" rows="3" readonly>${chain}</textarea>
            </div>
        `).join('');
    } else {
        chainContainer.innerHTML = `
            <div class="mb-2">
                <label class="form-label">Single Chain (${selectedSeq.sequence.length} residues):</label>
                <textarea class="form-control" rows="3" readonly>${selectedSeq.sequence}</textarea>
            </div>
        `;
    }

    chainDisplayContainer.style.display = 'block';
}

export function updateMembraneDisplay() {
    if (!state.currentSequence) return;

    const sequenceId = state.currentSequence.id;
    const regions = state.membraneRegions[sequenceId] || [];
    const chars = document.querySelectorAll('.sequence-char');

    chars.forEach(char => {
        char.classList.remove('selected', 'membrane-region');
    });

    regions.forEach(region => {
        for (let i = region.start; i <= region.end; i++) {
            const char = document.querySelector(`.sequence-char[data-position="${i}"]`);
            if (char) {
                char.classList.add('membrane-region');
            }
        }
    });
}

export function updateRegionCount() {
    if (!state.currentSequence) return;

    const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);
    const tokens = membraneSeqInput ? parseIdList((membraneSeqInput.value || '').toUpperCase()) : [];
    const regionCount = tokens.length;

    const countElement = document.getElementById(DOM.regionCount);
    if (countElement) {
        countElement.textContent = regionCount;

        const membraneCheckbox = document.getElementById('membrane');
        if (membraneCheckbox) {
            membraneCheckbox.checked = regionCount > 0;
            if (regionCount > 0) {
                membraneCheckbox.disabled = false;
            }
        }
    }
}

function parseIdList(value) {
    return Array.from(new Set(value
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0)));
}

export function renderStructuresList() {
    console.log('Rendering structures list:', state.structures);
    const structuresContainer = document.getElementById(DOM.structuresList);
    if (!structuresContainer) {
        console.warn('structuresList container not found');
        return;
    }

    if (state.structures && state.structures.length > 0) {
        const rows = state.structures.map((struct, index) => {
            const icon = struct.type === 'fasta' ? 'fa-file-lines' : 'fa-cube';
            const viewDisabled = !struct.file_url ? 'disabled' : '';

            // Show original filename for sequences from multi-seq FASTA
            const displayName = struct.original_filename && struct.original_filename !== struct.filename
                ? `${struct.filename} (${struct.original_filename})`
                : struct.filename;

            // Collect FP info from sequences
            const fpNames = new Set();
            if (struct.sequences) {
                struct.sequences.forEach(seq => {
                    if (seq.fp_name) fpNames.add(seq.fp_name);
                });
            }
            const fpBadge = fpNames.size > 0
                ? `<span class="badge bg-success ms-2" title="Fluorescent protein detected">🔬 ${Array.from(fpNames).join(', ')}</span>`
                : '';

            return `
            <div class="d-flex align-items-center justify-content-between py-1 border-bottom">
                <div class="form-check d-flex align-items-center">
                    <input class="form-check-input me-2" type="checkbox" ${struct.selected ? 'checked' : ''} onchange="toggleStructureSelected(${index})">
                    <label class="form-check-label">
                        <i class="fas ${icon} me-2"></i>
                        <span class="structure-name">${displayName}</span>
                        <small class="text-muted ms-2">(${formatFileSize(struct.file_size)})</small>
                        ${fpBadge}
                    </label>
                </div>
                <div class="btn-group btn-group-sm" role="group">
                    <button class="btn btn-outline-primary" onclick="showStructureByIndex(${index})" title="View 3D" ${viewDisabled}>
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-outline-danger" onclick="removeStructure(${index})" title="Remove">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>`;
        });
        structuresContainer.innerHTML = rows.join('');
    } else {
        structuresContainer.innerHTML = `
            <div class="text-muted text-center py-3">
                <i class="fas fa-upload fa-2x mb-2 opacity-50"></i>
                <div>No structures uploaded yet</div>
                <small>Upload FASTA files or PDB structures above</small>
            </div>
        `;
    }

    // Refresh dependent UI
    const allSequences = populateSequenceSelectorFromAllStructures();
    const selectedCount = (state.structures || []).filter(s => s.selected).length;
    const countElement = document.getElementById(DOM.selectedStructuresCount);
    if (countElement) {
        countElement.textContent = `${selectedCount} structure${selectedCount === 1 ? '' : 's'} selected`;
    }
    const selectorWrap = document.getElementById(DOM.sequenceSelector);
    if (selectorWrap) selectorWrap.style.display = allSequences.length > 0 ? 'block' : 'none';

    // Update multimer sequence display
    updateMultimerSequenceDisplay();
}

// Keep the old function for backward compatibility but redirect to new one
export function renderSelectedStructures() {
    renderStructuresList();
}
