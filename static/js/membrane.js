import { state } from './state.js';
import { DOM } from './config.js';
import { showInfo } from './ui.js';
import { updateMembraneDisplay, updateRegionCount } from './ui.js';
import { handleSequenceSelect } from './sequence.js';
import { saveState } from './persistence.js';
import { showStructurePreview } from './viewer.js';

let boundHandleMouseDown, boundHandleMouseMove, boundHandleMouseUp;

export function setupSequenceSelection() {
    const container = document.getElementById(DOM.sequenceText);
    if (!container) return;

    if (boundHandleMouseDown) {
        container.removeEventListener('mousedown', boundHandleMouseDown);
    }
    if (boundHandleMouseMove) {
        container.removeEventListener('mousemove', boundHandleMouseMove);
    }
    if (boundHandleMouseUp) {
        container.removeEventListener('mouseup', boundHandleMouseUp);
        container.removeEventListener('mouseleave', boundHandleMouseUp);
    }

    boundHandleMouseDown = handleMouseDown.bind(this);
    boundHandleMouseMove = handleMouseMove.bind(this);
    boundHandleMouseUp = handleMouseUp.bind(this);

    container.addEventListener('mousedown', boundHandleMouseDown);
    container.addEventListener('mousemove', boundHandleMouseMove);
    container.addEventListener('mouseup', boundHandleMouseUp);
    container.addEventListener('mouseleave', boundHandleMouseUp);

    container.addEventListener('selectstart', (e) => {
        if (state.isSelecting) {
            e.preventDefault();
        }
    });

    const clearBtn = document.getElementById(DOM.clearRegions);
    if (clearBtn) {
        clearBtn.onclick = () => clearMembraneRegions();
    }

    updateRegionCount();
}

function handleMouseDown(e) {
    const char = e.target.closest('.sequence-char');
    if (!char) return;

    e.preventDefault();
    state.isSelecting = true;
    state.startPos = parseInt(char.dataset.position);
    highlightSelection(state.startPos, state.startPos);
}

function handleMouseMove(e) {
    if (!state.isSelecting) return;

    const char = e.target.closest('.sequence-char');
    if (!char) return;

    const endPos = parseInt(char.dataset.position);
    highlightSelection(Math.min(state.startPos, endPos), Math.max(state.startPos, endPos));
}

function handleMouseUp(e) {
    if (!state.isSelecting) return;

    const char = e.target.closest('.sequence-char');
    if (char) {
        const endPos = parseInt(char.dataset.position);
        const start = Math.min(state.startPos, endPos);
        const end = Math.max(state.startPos, endPos);

        if (start !== end) {
            addMembraneRegion(start, end);
        }
    }

    state.isSelecting = false;
    state.startPos = -1;
    updateMembraneDisplay();
    highlightFromTextField();
}

function highlightSelection(start, end) {
    const chars = document.querySelectorAll('.sequence-char');
    chars.forEach((char, i) => {
        if (i >= start && i <= end) {
            char.classList.add('selected');
        } else {
            char.classList.remove('selected');
        }
    });
}

function addMembraneRegion(start, end) {
    if (start === end) return;
    if (!state.currentSequence) return;

    const sequenceId = state.currentSequence.id;

    if (!state.membraneRegions[sequenceId]) {
        state.membraneRegions[sequenceId] = [];
    }

    const overlapping = state.membraneRegions[sequenceId].some(region =>
        (start >= region.start && start <= region.end) ||
        (end >= region.start && end <= region.end) ||
        (start <= region.start && end >= region.end)
    );

    if (!overlapping) {
        state.membraneRegions[sequenceId].push({ start, end });
        state.membraneRegions[sequenceId].sort((a, b) => a.start - b.start);

        updateMembraneDisplay();
        updateRegionCount();
        saveState();

        showInfo(`Added membrane region: ${start + 1}-${end + 1}`);

        const aaSegment = state.currentSequence.sequence.slice(start, end + 1);
        updateMembraneInputWithAASegment(aaSegment);
        highlightFromTextField();
        updateRegionCount();
        update3DViewerWithMembrane();
    } else {
        showInfo('Region overlaps with existing membrane region', 'warning');
    }
}

export function clearMembraneRegions() {
    if (!state.currentSequence) return;

    const sequenceId = state.currentSequence.id;
    if (state.membraneRegions[sequenceId]) {
        state.membraneRegions[sequenceId] = [];
        updateMembraneDisplay();
        saveState();
        const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);
        if (membraneSeqInput) {
            membraneSeqInput.value = '';
        }
        updateRegionCount();
        highlightFromTextField();
        updateSelectorHighlightFromTextField();

        const membraneCheckbox = document.getElementById('membrane');
        if (membraneCheckbox) {
            membraneCheckbox.checked = false;
        }

        showInfo('Cleared all membrane regions');
        update3DViewerWithMembrane();
    }
}

export function handleMembraneSeqInputEvent(event, forceApply = false) {
    const inputEl = event.target;
    const select = document.getElementById(DOM.sequenceSelect);
    if (!inputEl || !select) return;

    inputEl.value = (inputEl.value || '').toUpperCase();
    const fullList = parseIdList(inputEl.value);
    const raw = inputEl.value;
    const lastToken = raw.split(',').pop().trim();
    if (!lastToken) {
        inputEl.classList.remove('is-invalid', 'is-valid');
        updateRegionCount();
        return;
    }

    const candidates = [];
    (state.uploadedSequences || []).forEach(seqObj => {
        if ((seqObj.sequence || '').toUpperCase() === lastToken) {
            candidates.push({ type: 'exact', parentId: seqObj.id });
        }
        const chains = (seqObj.sequence || '').split(':');
        if (chains.length > 1) {
            chains.forEach(ch => {
                if ((ch || '').toUpperCase() === lastToken) {
                    candidates.push({ type: 'exact_chain', parentId: seqObj.id });
                }
            });
        }
    });

    if (candidates.length === 1) {
        select.value = candidates[0].parentId;
        handleSequenceSelect();
        inputEl.classList.remove('is-invalid');
        inputEl.classList.add('is-valid');
        highlightFromTextField();
        updateSelectorHighlightFromTextField();
        updateRegionCount();
        return;
    }

    if (candidates.length === 0) {
        const partialParents = new Set();
        (state.uploadedSequences || []).forEach(seqObj => {
            const seqLower = (seqObj.sequence || '').toLowerCase();
            const tokLower = lastToken.toLowerCase();
            if (seqLower.includes(tokLower)) partialParents.add(seqObj.id);
            const chains = (seqObj.sequence || '').split(':');
            if (chains.length > 1) {
                chains.forEach(ch => {
                    if (ch.toLowerCase().includes(tokLower)) partialParents.add(seqObj.id);
                });
            }
        });
        if (partialParents.size === 1) {
            select.value = Array.from(partialParents)[0];
            if (forceApply) handleSequenceSelect();
            inputEl.classList.remove('is-invalid');
            inputEl.classList.add('is-valid');
            highlightFromTextField();
            updateSelectorHighlightFromTextField();
            updateRegionCount();
            return;
        }
    }

    const overallMatched = matchedParentIdsFromTokens(fullList);
    if (overallMatched.size === 1) {
        const onlyId = Array.from(overallMatched)[0];
        if (select.value !== onlyId) {
            select.value = onlyId;
            handleSequenceSelect();
        }
        inputEl.classList.remove('is-invalid');
        inputEl.classList.add('is-valid');
    }

    inputEl.classList.remove('is-valid');
    inputEl.classList.add('is-invalid');
    highlightFromTextField();
    updateSelectorHighlightFromTextField();
    updateRegionCount();
    update3DViewerWithMembrane();
}

export function highlightFromTextField() {
    if (!state.currentSequence) return;
    const sequenceText = document.getElementById(DOM.sequenceText);
    if (!sequenceText) return;

    const chars = sequenceText.querySelectorAll('.sequence-char');
    chars.forEach(el => el.classList.remove('selected'));

    const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);
    if (!membraneSeqInput) return;
    const tokens = parseIdList((membraneSeqInput.value || '').toUpperCase());
    if (tokens.length === 0) return;

    const fullSeq = (state.currentSequence.sequence || '');
    const fullSeqLower = fullSeq.toLowerCase();
    tokens.forEach(token => {
        if (!token) return;
        const tokenLower = token.toLowerCase();
        let startIndex = 0;
        while (startIndex <= fullSeqLower.length - tokenLower.length) {
            const idx = fullSeqLower.indexOf(tokenLower, startIndex);
            if (idx === -1) break;
            for (let p = idx; p < idx + token.length; p++) {
                const span = sequenceText.querySelector(`.sequence-char[data-position="${p}"]`);
                if (span) span.classList.add('selected');
            }
            startIndex = idx + tokenLower.length;
        }
    });
}

function updateMembraneInputWithAASegment(aaSegment) {
    const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);
    if (!membraneSeqInput || !aaSegment) return;
    const list = parseIdList(membraneSeqInput.value);
    const segUpper = (aaSegment || '').toUpperCase();
    if (!list.includes(segUpper)) {
        list.push(segUpper);
        membraneSeqInput.value = joinIdList(list);
    }
    membraneSeqInput.classList.remove('is-invalid');
    membraneSeqInput.classList.add('is-valid');
    setTimeout(() => membraneSeqInput.classList.remove('is-valid'), 800);
}

export function updateSelectorHighlightFromTextField() {
    const select = document.getElementById(DOM.sequenceSelect);
    const inputEl = document.getElementById(DOM.membraneSeqInput);
    if (!select || !inputEl) return;
    const tokens = parseIdList((inputEl.value || '').toUpperCase());
    const matched = matchedParentIdsFromTokens(tokens);

    Array.from(select.options).forEach(opt => {
        if (!opt.dataset.label) opt.dataset.label = opt.textContent;
        const original = opt.dataset.label;
        if (matched.has(opt.value)) {
            if (!original.startsWith('★ ')) {
                opt.textContent = `★ ${original}`;
            } else {
                opt.textContent = original;
            }
            opt.classList.add('matched-option');
            opt.style.backgroundColor = '#fff3cd';
            opt.style.fontWeight = '600';
        } else {
            opt.textContent = original.replace(/^★\s+/, '');
            opt.classList.remove('matched-option');
            opt.style.backgroundColor = '';
            opt.style.fontWeight = '';
        }
    });
}

function matchedParentIdsFromTokens(tokens) {
    const matched = new Set();
    const seqs = state.uploadedSequences || [];
    tokens.forEach(token => {
        if (!token) return;
        const tokLower = token.toLowerCase();
        seqs.forEach(seqObj => {
            const full = seqObj.sequence;
            if (full === token || full.toLowerCase().includes(tokLower)) {
                matched.add(seqObj.id);
                return;
            }
            const chains = full.split(':');
            if (chains.length > 1) {
                for (const ch of chains) {
                    if (ch === token || ch.toLowerCase().includes(tokLower)) {
                        matched.add(seqObj.id);
                        break;
                    }
                }
            }
        });
    });
    return matched;
}

function parseIdList(value) {
    return Array.from(new Set(value
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0)));
}

function joinIdList(list) {
    return list.join(', ');
}

// Update the 3D viewer with membrane regions
let updateViewerTimeout = null;

export async function update3DViewerWithMembrane() {
    // Debounce to avoid frequent reloads during typing
    if (updateViewerTimeout) {
        clearTimeout(updateViewerTimeout);
    }

    updateViewerTimeout = setTimeout(async () => {
        console.log('Updating 3D viewer with membrane regions...');
        if (!state.currentSequence) {
            console.log('No current sequence for 3D update');
            return;
        }

        // Find structure
        let structure = null;
        if (state.currentSequence.structureIndex !== undefined && state.structures) {
            structure = state.structures[state.currentSequence.structureIndex];
        }

        if (!structure || !structure.file_url) {
            console.log('No linked structure found for 3D update');
            return;
        }

        // Prepare membrane regions for all sequences in the structure
        // We need to check both state.membraneRegions AND the text input matches

        // 1. Parse text input for global matches
        const membraneSeqInput = document.getElementById(DOM.membraneSeqInput);
        const tokens = membraneSeqInput ? parseIdList((membraneSeqInput.value || '').toUpperCase()) : [];


        const viewerSequences = structure.sequences.map(seq => {
            // Reconstruct fullId used in membraneRegions
            const fullId = `${structure.filename}_${seq.id}`;

            // Start with manual regions
            const regions = [...(state.membraneRegions[fullId] || [])];

            // Add matches from text input
            if (tokens.length > 0) {
                const seqLower = (seq.sequence || '').toLowerCase();
                tokens.forEach(token => {
                    if (!token) return;
                    const tokenLower = token.toLowerCase();
                    let startIndex = 0;
                    while (startIndex <= seqLower.length - tokenLower.length) {
                        const idx = seqLower.indexOf(tokenLower, startIndex);
                        if (idx === -1) break;

                        // Avoid duplicates with manual regions
                        const isDuplicate = regions.some(r => r.start === idx && r.end === idx + token.length - 1);
                        if (!isDuplicate) {
                            regions.push({ start: idx, end: idx + token.length - 1 });
                        }

                        startIndex = idx + tokenLower.length;
                    }
                });
            }

            // Create a copy of existing FPs (if any)
            const fps = [...(seq.fps || [])];

            // Add membrane regions as "FPs" with yellow color
            regions.forEach(reg => {
                fps.push({
                    name: 'Membrane',
                    start: reg.start,
                    end: reg.end,
                    color: 'yellow'
                });
            });

            return {
                ...seq,
                fps: fps
            };
        });

        // Deep inspection of the data being passed
        const debugData = viewerSequences.map(s => ({
            id: s.id,
            chain: s.chain,
            fps_count: s.fps ? s.fps.length : 0,
            regions: s.fps ? s.fps.map(f => `${f.name}:${f.start}-${f.end}(${f.color})`).join(', ') : 'none'
        }));
        console.log('DEBUG: update3DViewerWithMembrane constructed sequences:', JSON.stringify(debugData, null, 2));

        try {
            await showStructurePreview(structure.file_url, structure.filename, viewerSequences);
        } catch (e) {
            console.error('Error updating 3D viewer:', e);
        }
    }, 500); // 500ms delay
}
