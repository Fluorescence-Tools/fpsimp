import { state } from './state.js';
import { DOM } from './config.js';
import { displayChainInformation, updateMembraneDisplay, updateRegionCount } from './ui.js';
import { setupSequenceSelection, highlightFromTextField } from './membrane.js';

export function handleSequenceSelect() {
    console.log('handleSequenceSelect called');
    const select = document.getElementById(DOM.sequenceSelect);
    const sequenceId = select ? select.value : null;
    const sequenceDisplay = document.getElementById(DOM.sequenceDisplay);
    const sequenceText = document.getElementById(DOM.sequenceText);

    if (!sequenceId) {
        console.log('No sequenceId, returning');
        return;
    }
    
    console.log('Processing sequenceId:', sequenceId);

    // Get all available sequences (from FASTA upload or structures)
    let allSequences = [];
    let seq = null;

    // First try to find in uploaded sequences (FASTA)
    if (state.uploadedSequences && state.uploadedSequences.length > 0) {
        seq = state.uploadedSequences.find(s => s.id === sequenceId);
        allSequences = state.uploadedSequences;
    }

    // If not found, try to find in structure sequences
    if (!seq && state.structures && state.structures.length > 0) {
        state.structures.forEach((structure, structIndex) => {
            if (structure.sequences) {
                structure.sequences.forEach(structSeq => {
                    const fullId = `${structure.filename}_${structSeq.id}`;
                    const sequenceWithMeta = {
                        ...structSeq,
                        id: fullId,
                        structureIndex: structIndex,
                        structureName: structure.filename,
                        originalId: structSeq.id
                    };
                    allSequences.push(sequenceWithMeta);
                    if (fullId === sequenceId) {
                        seq = sequenceWithMeta;
                    }
                });
            }
        });
    }

    if (!seq) return;

    state.currentSequence = seq;

    if (sequenceDisplay) sequenceDisplay.style.display = 'block';

    if (!state.membraneRegions[sequenceId]) {
        state.membraneRegions[sequenceId] = [];
    }

    if (sequenceText) {
        populateSequenceText(sequenceText, seq.sequence);
    }

    displayChainInformation(sequenceId, allSequences);

    setupSequenceSelection();
    updateMembraneDisplay();
    updateRegionCount();

    highlightFromTextField();
}

function populateSequenceText(sequenceText, sequenceToDisplay) {
    sequenceText.innerHTML = '';
    
    const wrapper = document.createElement('div');
    wrapper.className = 'sequence-wrapper';
    
    // Detect FP region for highlighting
    let fpStart = -1;
    let fpEnd = -1;
    let fpColor = null;
    
    console.log('FP Detection Debug:', {
        hasCurrentSequence: !!state.currentSequence,
        fp_name: state.currentSequence?.fp_name,
        fp_color: state.currentSequence?.fp_color,
        fp_match_type: state.currentSequence?.fp_match_type,
        fp_motif: state.currentSequence?.fp_motif
    });
    
    if (state.currentSequence && state.currentSequence.fp_name) {
        // Use backend-calculated FP boundaries if available
        if (state.currentSequence.fp_start !== undefined && state.currentSequence.fp_end !== undefined) {
            fpStart = state.currentSequence.fp_start;
            fpEnd = state.currentSequence.fp_end;
            console.log('FP Using backend boundaries:', { fpStart, fpEnd });
        } else if (state.currentSequence.fp_match_type === 'full') {
            fpStart = 0;
            fpEnd = sequenceToDisplay.length - 1;
            console.log('FP Full match detected, highlighting entire sequence');
        } else if (state.currentSequence.fp_match_type === 'motif' && state.currentSequence.fp_motif) {
            // Fallback to motif-only highlighting
            const seq = sequenceToDisplay.toUpperCase();
            const motifPos = seq.indexOf(state.currentSequence.fp_motif.toUpperCase());
            console.log('FP Motif search:', {
                motif: state.currentSequence.fp_motif,
                motifUpper: state.currentSequence.fp_motif.toUpperCase(),
                motifPos: motifPos,
                seqStart: seq.substring(0, 50)
            });
            if (motifPos >= 0) {
                fpStart = motifPos;
                fpEnd = motifPos + state.currentSequence.fp_motif.length - 1;
                console.log(`FP Motif found at position ${fpStart}-${fpEnd}`);
            }
        }
        
        // Set color based on FP type
        fpColor = state.currentSequence.fp_color || 'green';
        console.log('FP highlighting range:', { fpStart, fpEnd, fpColor });
    }
    
    const charsPerLine = 50;
    
    for (let i = 0; i < sequenceToDisplay.length; i += charsPerLine) {
        const line = document.createElement('div');
        line.className = 'sequence-line';
        
        const pos = document.createElement('span');
        pos.className = 'position-indicator';
        pos.textContent = (i + 1).toString().padStart(4, ' ');
        line.appendChild(pos);
        
        for (let j = 0; j < charsPerLine && (i + j) < sequenceToDisplay.length; j++) {
            const position = i + j;
            const char = document.createElement('span');
            char.className = 'sequence-char';
            char.textContent = sequenceToDisplay[position];
            char.dataset.position = position.toString();
            
            // Apply FP highlighting if position is within FP region
            if (fpStart >= 0 && position >= fpStart && position <= fpEnd) {
                char.classList.add('fp-region');
                if (fpColor === 'green') {
                    char.classList.add('fp-green');
                } else if (fpColor === 'magenta') {
                    char.classList.add('fp-magenta');
                } else if (fpColor === 'yellow') {
                    char.classList.add('fp-yellow');
                }
            }
            
            line.appendChild(char);
        }
        
        wrapper.appendChild(line);
    }
    
    sequenceText.appendChild(wrapper);
    
    setTimeout(() => {
        setupSequenceSelection();
        updateMembraneDisplay();
        updateRegionCount();
    }, 0);
}
