import { state } from './state.js';
import { DOM } from './config.js';
import { saveState } from './persistence.js';

export function toggleFPSiteSelection(enabled) {
    const fpSiteSelection = document.getElementById(DOM.fpSiteSelection);
    if (fpSiteSelection) {
        fpSiteSelection.style.display = enabled ? 'block' : 'none';
    }
}

export function setupFPSiteListeners() {
    ['donor', 'acceptor'].forEach(sitePrefix => {
        [1, 2].forEach(num => {
            const chainSelect = document.getElementById(`${sitePrefix}Chain${num}`);
            const positionInput = document.getElementById(`${sitePrefix}Pos${num}`);

            if (chainSelect) {
                chainSelect.addEventListener('change', () => {
                    // Only auto-populate if we're setting chain 1
                    if (num === 1) {
                        autoPopulateFPSites(chainSelect.value, sitePrefix);
                    }
                    updateFPSiteSummary();
                    // Update position input max value based on selected chain
                    updatePositionInputMax(`${sitePrefix}Pos${num}`, chainSelect.value);
                });
            }
            if (positionInput) {
                positionInput.addEventListener('input', () => {
                    updateFPSiteSummary();
                    // Validate position is within sequence bounds
                    validatePositionInput(positionInput, chainSelect?.value);
                });
                // Set initial max value
                updatePositionInputMax(`${sitePrefix}Pos${num}`, chainSelect?.value);
            }
        });
    });

    // Check initial state of measure checkbox on load
    const measureCheckbox = document.getElementById('measure');
    if (measureCheckbox) {
        toggleFPSiteSelection(measureCheckbox.checked);
    }
}

/**
 * Automatically populates FP site positions for all recognized FPs in the uploaded sequences.
 * It assigns the first FP found to FP1 and the second different FP (or different chain of same FP) to FP2.
 * Handles multiple FPs within a single sequence.
 */
export function fullyAutoPopulateFPs() {
    console.log('fullyAutoPopulateFPs called', state.uploadedSequences?.length, 'sequences');
    if (!state.uploadedSequences || state.uploadedSequences.length === 0) return;

    // Collect all FPs from all sequences, handling multiple FPs per sequence
    const allFPs = [];
    
    state.uploadedSequences.forEach((sequence, seqIndex) => {
        // Check for multiple FPs in sequence
        if (sequence.fps && sequence.fps.length > 0) {
            console.log(`Found ${sequence.fps.length} FPs in sequence ${seqIndex}: ${sequence.fps.map(fp => fp.name).join(', ')}`);
            sequence.fps.forEach((fp, fpIndex) => {
                if (fp.dipole_triplets && fp.dipole_triplets.length >= 2) {
                    allFPs.push({
                        seqIndex: seqIndex,
                        fpIndex: fpIndex,
                        fpName: fp.name,
                        fpColor: fp.color,
                        dipole_triplets: fp.dipole_triplets,
                        sequence: sequence
                    });
                }
            });
        }
        // Check for single FP (legacy support)
        else if (sequence.fp_name && sequence.dipole_triplets && sequence.dipole_triplets.length >= 2) {
            console.log(`Found single FP in sequence ${seqIndex}: ${sequence.fp_name}`);
            allFPs.push({
                seqIndex: seqIndex,
                fpIndex: 0,
                fpName: sequence.fp_name,
                fpColor: sequence.fp_color,
                dipole_triplets: sequence.dipole_triplets,
                sequence: sequence
            });
        }
    });

    if (allFPs.length === 0) {
        console.log('No valid FPs found for auto-population');
        return;
    }

    console.log(`Found ${allFPs.length} total FPs for auto-population`);

    // Assign first FP to FP1 (donor)
    const fp1 = allFPs[0];
    autoPopulateFPSitesFromFP(fp1, 'donor');

    // Assign second FP to FP2 (acceptor) if available
    if (allFPs.length >= 2) {
        const fp2 = allFPs[1];
        autoPopulateFPSitesFromFP(fp2, 'acceptor');
    } else {
        console.log('Only one FP found, FP2 will need manual selection');
    }
}

/**
 * Automatically populates FP site positions from a specific FP object
 * @param {Object} fpObj - FP object containing seqIndex, fpIndex, fpName, dipole_triplets, etc.
 * @param {string} sitePrefix - 'donor' or 'acceptor'
 */
export function autoPopulateFPSitesFromFP(fpObj, sitePrefix) {
    console.log(`autoPopulateFPSitesFromFP called for ${sitePrefix}, FP=${fpObj.fpName}, seqIndex=${fpObj.seqIndex}`);
    
    const chainIndexStr = fpObj.seqIndex.toString();
    const sequence = fpObj.sequence;
    const dipoleTriplets = fpObj.dipole_triplets;

    if (!sequence || !dipoleTriplets || dipoleTriplets.length < 2) {
        console.log(`Invalid FP data for ${fpObj.fpName}`);
        return;
    }

    console.log(`Auto-populating sites for recognized FP: ${fpObj.fpName} on ${sitePrefix}`);

    // Set the chain selector value first
    const id1 = `${sitePrefix}Chain1`;
    const id2 = `${sitePrefix}Chain2`;
    const chain1 = document.getElementById(id1);
    const chain2 = document.getElementById(id2);
    console.log(`Setting ${id1} and ${id2} to ${chainIndexStr}`);
    if (chain1) chain1.value = chainIndexStr;
    if (chain2) chain2.value = chainIndexStr;

    const seq = sequence.sequence.toUpperCase();
    const positions = [];

    dipoleTriplets.forEach(triplet => {
        const tripletUpper = triplet.toUpperCase();
        const index = seq.indexOf(tripletUpper);
        if (index !== -1) {
            // Central AA is at index + 1 in 0-based, so index + 2 in 1-based
            positions.push(index + 2);
        }
    });

    if (positions.length >= 2) {
        const pos1 = document.getElementById(`${sitePrefix}Pos1`);
        const pos2 = document.getElementById(`${sitePrefix}Pos2`);
        const measureCheckbox = document.getElementById('measure');

        if (chain1 && pos1) {
            chain1.value = chainIndexStr;
            pos1.value = positions[0];
        }
        if (chain2 && pos2) {
            chain2.value = chainIndexStr; // Use same chain for both sites by default
            pos2.value = positions[1];
        }

        // Enable distance computation automatically
        if (measureCheckbox && !measureCheckbox.checked) {
            measureCheckbox.checked = true;
            // Trigger change event to show the selection div
            measureCheckbox.dispatchEvent(new Event('change'));
        } else {
            // If already checked, just make sure it's visible
            toggleFPSiteSelection(true);
        }

        updateFPSiteSummary();
        console.log(`Successfully populated ${fpObj.fpName} sites: ${positions[0]}, ${positions[1]}`);
    } else {
        console.warn(`Could not find enough dipole triplet positions for ${fpObj.fpName}`);
    }
}

/**
 * Automatically populates FP site positions if the selected chain is a recognized FP
 * @param {string} chainIndexStr - The index of the sequence in state.uploadedSequences
 * @param {string} sitePrefix - 'donor' or 'acceptor'
 */
export function autoPopulateFPSites(chainIndexStr, sitePrefix) {
    console.log(`autoPopulateFPSites called for ${sitePrefix}, chainIndexStr=${chainIndexStr}`);
    if (!chainIndexStr) return;

    const chainIndex = parseInt(chainIndexStr);
    const sequence = state.uploadedSequences[chainIndex];

    if (sequence && sequence.fp_name && sequence.dipole_triplets && sequence.dipole_triplets.length >= 2) {
        console.log(`Auto-populating sites for recognized FP: ${sequence.fp_name} on ${sitePrefix}`);

        // Set the chain selector value first
        const id1 = `${sitePrefix}Chain1`;
        const id2 = `${sitePrefix}Chain2`;
        const chain1 = document.getElementById(id1);
        const chain2 = document.getElementById(id2);
        console.log(`Setting ${id1} and ${id2} to ${chainIndexStr}`);
        if (chain1) chain1.value = chainIndexStr;
        if (chain2) chain2.value = chainIndexStr;

        const seq = sequence.sequence.toUpperCase();
        const triplets = sequence.dipole_triplets;
        const positions = [];

        triplets.forEach(triplet => {
            const tripletUpper = triplet.toUpperCase();
            const index = seq.indexOf(tripletUpper);
            if (index !== -1) {
                // Central AA is index + 1
                // sequence positions are 1-based, so its index + 1 + 1 (for 1-based)
                // Wait, index is 0-based. index + 1 is second char position (1-based index).
                // Let's be precise: index is 0-based start of triplet. 
                // Central AA is at index + 1 in 0-based.
                // In 1-based it is (index + 1) + 1.
                positions.push(index + 2);
            }
        });

        if (positions.length >= 2) {
            const chain1 = document.getElementById(`${sitePrefix}Chain1`);
            const pos1 = document.getElementById(`${sitePrefix}Pos1`);
            const chain2 = document.getElementById(`${sitePrefix}Chain2`);
            const pos2 = document.getElementById(`${sitePrefix}Pos2`);
            const measureCheckbox = document.getElementById('measure');

            if (chain1 && pos1) {
                chain1.value = chainIndexStr;
                pos1.value = positions[0];
            }
            if (chain2 && pos2) {
                chain2.value = chainIndexStr; // Use same chain for both sites by default
                pos2.value = positions[1];
            }

            // Enable distance computation automatically
            if (measureCheckbox && !measureCheckbox.checked) {
                measureCheckbox.checked = true;
                // Trigger change event to show the selection div
                measureCheckbox.dispatchEvent(new Event('change'));
            } else {
                // If already checked, just make sure it's visible
                toggleFPSiteSelection(true);
            }

            updateFPSiteSummary();
        }
    }
}

function updateFPSiteSummary() {
    updateFPSites();
    const summary = document.getElementById(DOM.fpSiteSummary);
    if (!summary) return;

    const parts = [];

    if (state.fpSites.donor.site1) {
        const site1 = state.fpSites.donor.site1;
        parts.push(`FP1: ${site1.chainId}:${site1.aminoAcid}${site1.position}`);

        if (state.fpSites.donor.site2) {
            const site2 = state.fpSites.donor.site2;
            parts[parts.length - 1] += ` + ${site2.chainId}:${site2.aminoAcid}${site2.position}`;
        }
    }

    if (state.fpSites.acceptor.site1) {
        const site1 = state.fpSites.acceptor.site1;
        parts.push(`FP2: ${site1.chainId}:${site1.aminoAcid}${site1.position}`);

        if (state.fpSites.acceptor.site2) {
            const site2 = state.fpSites.acceptor.site2;
            parts[parts.length - 1] += ` + ${site2.chainId}:${site2.aminoAcid}${site2.position}`;
        }
    }

    if (parts.length === 0) {
        summary.textContent = 'No sites selected';
    } else {
        const donorSites = state.fpSites.donor.site1 ? (state.fpSites.donor.site2 ? 2 : 1) : 0;
        const acceptorSites = state.fpSites.acceptor.site1 ? (state.fpSites.acceptor.site2 ? 2 : 1) : 0;
        const measurementType = (donorSites === 2 && acceptorSites === 2) ? 'distances + orientations' : 'distances only';

        summary.innerHTML = `${parts.join(' | ')} <span class="badge bg-info">${measurementType}</span>`;
    }
}

function updateFPSites() {
    state.fpSites = {
        donor: {
            site1: getFPSite('donorChain1', 'donorPos1'),
            site2: getFPSite('donorChain2', 'donorPos2')
        },
        acceptor: {
            site1: getFPSite('acceptorChain1', 'acceptorPos1'),
            site2: getFPSite('acceptorChain2', 'acceptorPos2')
        }
    };
    saveState();
}

function getFPSite(chainId, posId) {
    const chainSelect = document.getElementById(chainId);
    const posInput = document.getElementById(posId);

    if (chainSelect && posInput && chainSelect.value !== '' && posInput.value !== '') {
        const chainIndex = parseInt(chainSelect.value);
        const position = parseInt(posInput.value);
        const sequence = state.uploadedSequences[chainIndex];

        if (sequence && position >= 1 && position <= sequence.sequence.length) {
            return {
                chainIndex: chainIndex,
                chainId: sequence.chain || sequence.id,
                position: position,
                aminoAcid: sequence.sequence[position - 1]
            };
        }
    }
    return null;
}

export function populateFPChainSelectors() {
    const chainSelectors = ['donorChain1', 'donorChain2', 'acceptorChain1', 'acceptorChain2'];
    console.log('Populating FP chain selectors with', state.uploadedSequences?.length || 0, 'sequences');

    chainSelectors.forEach(selectorId => {
        const selector = document.getElementById(selectorId);
        if (selector && state.uploadedSequences) {
            const currentValue = selector.value;
            selector.innerHTML = '<option value="">Select Chain</option>';

            state.uploadedSequences.forEach((seq, index) => {
                const option = document.createElement('option');
                option.value = index;

                // Use originalId and structureName for more readable options
                let label = `${seq.originalId} (${seq.sequence.length} aa) from ${seq.structureName}`;
                if (seq.fp_name) {
                    label = `🔬 ${seq.fp_name} - ${label}`;
                }

                option.textContent = label;
                selector.appendChild(option);
            });

            // Try to restore previous value if it still exists
            if (currentValue && Array.from(selector.options).some(opt => opt.value === currentValue)) {
                selector.value = currentValue;
            }
            
            // Update corresponding position input max value
            const posInputId = selectorId.replace('Chain', 'Pos');
            updatePositionInputMax(posInputId, currentValue);
        }
    });
}

// Update position input max value based on selected chain
function updatePositionInputMax(posInputId, chainIndexStr) {
    const posInput = document.getElementById(posInputId);
    if (!posInput) return;
    
    if (chainIndexStr && state.uploadedSequences) {
        const chainIndex = parseInt(chainIndexStr);
        const sequence = state.uploadedSequences[chainIndex];
        
        if (sequence) {
            posInput.max = sequence.sequence.length;
            posInput.placeholder = `AA Position (1-${sequence.sequence.length})`;
            posInput.title = `Enter residue index from 1 to ${sequence.sequence.length}`;
        }
    } else {
        // No chain selected
        posInput.removeAttribute('max');
        posInput.placeholder = 'AA Position';
        posInput.title = 'Select a chain first, then enter residue index';
    }
}

// Validate position input is within sequence bounds
function validatePositionInput(posInput, chainIndexStr) {
    if (!chainIndexStr || !state.uploadedSequences) return;
    
    const chainIndex = parseInt(chainIndexStr);
    const sequence = state.uploadedSequences[chainIndex];
    const position = parseInt(posInput.value);
    
    if (sequence && position) {
        if (position < 1 || position > sequence.sequence.length) {
            posInput.setCustomValidity(`Position must be between 1 and ${sequence.sequence.length}`);
            posInput.classList.add('is-invalid');
        } else {
            posInput.setCustomValidity('');
            posInput.classList.remove('is-invalid');
            posInput.classList.add('is-valid');
        }
    } else {
        posInput.setCustomValidity('');
        posInput.classList.remove('is-invalid', 'is-valid');
    }
}
