import { state } from './state.js';
import { DOM } from './config.js';
import { showError, showSuccess, setLoadingState } from './ui.js';

function deepCopyAndClean(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => deepCopyAndClean(item));
    }

    const cleanObj = {};
    for (const [key, value] of Object.entries(obj)) {
        // Skip functions and undefined values
        if (typeof value === 'function' || value === undefined) {
            continue;
        }

        // Recursively clean nested objects
        cleanObj[key] = deepCopyAndClean(value);
    }

    return cleanObj;
}

async function submitJobApi(jobData) {
    // Create a clean, serializable copy of the data
    const cleanJobData = deepCopyAndClean({
        upload_id: jobData.upload_id,
        parameters: {
            ...jobData.parameters,
            // Ensure membrane_regions is a proper array of plain objects
            membrane_regions: (jobData.parameters.membrane_regions || []).map(region => ({
                start: Number(region.start),
                end: Number(region.end)
            }))
        }
    });

    // Debug: Log the cleaned data
    console.log('Submitting job with cleaned data:', JSON.parse(JSON.stringify(cleanJobData)));

    try {
        const response = await fetch('/api/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(cleanJobData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server responded with error:', errorText);
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error in submitJobApi:', error);
        throw error;
    }
}

async function fetchStatus(jobId) {
    const response = await fetch(`/api/status/${jobId}`);
    return response.json();
}

async function fetchResults(jobId) {
    const response = await fetch(`/api/results/${jobId}`);
    return response.json();
}

export async function submitJob(event) {
    event.preventDefault();
    console.group('submitJob');

    // Check if we have any selected structures
    const selectedStructures = state.getSelectedStructures();
    if (selectedStructures.length === 0) {
        const errorMsg = 'Please upload and select at least one structure for multimer assembly';
        console.error(errorMsg);
        showError(errorMsg);
        console.groupEnd();
        return;
    }

    // Use the upload_id from the first selected structure (for backward compatibility)
    // In a full multimer implementation, we'd pass all selected structure IDs
    const primaryStructure = selectedStructures[0];
    const uploadId = primaryStructure.upload_id;
    console.log('Selected structures:', selectedStructures.length);
    console.log('Primary upload ID:', uploadId);

    // Use multimer sequence ID for membrane region tracking when multiple FASTA structures
    const fastaStructures = selectedStructures.filter(s => s.type === 'fasta');
    let sequenceId = null;

    if (fastaStructures.length > 1) {
        // Multiple FASTA structures - use multimer sequence
        sequenceId = 'multimer_sequence';
    } else if (fastaStructures.length === 1) {
        // Single FASTA structure - use original sequence ID format
        const structure = fastaStructures[0];
        if (structure.sequences && structure.sequences.length > 0) {
            sequenceId = `${structure.filename}_${structure.sequences[0].id}`;
        }
    } else if (selectedStructures.length > 0) {
        // PDB structures - use first structure's sequence
        const firstStructure = selectedStructures[0];
        if (firstStructure.sequences && firstStructure.sequences.length > 0) {
            sequenceId = `${firstStructure.filename}_${firstStructure.sequences[0].id}`;
        }
    }
    console.log('Using sequence ID for membrane regions:', sequenceId);

    console.log('Collecting parameters...');
    const parameters = collectParameters(sequenceId);
    console.log('Collected parameters:', parameters);

    // Get email for notifications
    const emailInput = document.getElementById('emailInput');
    const email = emailInput ? emailInput.value.trim() : '';

    const jobData = {
        upload_id: uploadId,
        parameters: parameters,
        email: email
    };
    console.log('Prepared job data:', jobData);

    setLoadingState(true);

    try {
        console.log('Submitting job...');
        const data = await submitJobApi(jobData);
        console.log('Server response:', data);

        if (data.job_id) {
            console.log('Job submitted successfully with ID:', data.job_id);
            state.currentJobId = data.job_id;
            state.lastJobResultsUrl = null; // Clear previous results

            // Save state to persist job ID
            const { saveState } = await import('./persistence.js');
            saveState();

            showJobStatus(data.job_id, data.results_url);
            startStatusPolling();
            showSuccess('Job submitted successfully!');
        } else {
            const errorMsg = data.error || 'No job ID returned from server';
            console.error('Error from server:', errorMsg);
            showError(errorMsg);
        }
    } catch (error) {
        const errorMsg = 'Submission failed: ' + (error.message || 'Unknown error');
        console.error(errorMsg, error);
        console.error('Error stack:', error.stack);
        showError(errorMsg);
    } finally {
        setLoadingState(false);
        console.groupEnd();
    }
}

function collectParameters(sequenceId) {
    console.log('Collecting parameters for sequence:', sequenceId);

    const val = (id, def = null) => {
        try {
            const el = document.getElementById(id);
            if (!el) {
                console.log(`Element with ID '${id}' not found, using default:`, def);
                return def;
            }
            const value = el.type === 'checkbox' ? el.checked : (el.value === '' ? def : el.value);
            console.log(`Value for ${id} (type: ${el.type}):`, value);
            return value;
        } catch (error) {
            console.error(`Error getting value for ${id}:`, error);
            return def;
        }
    };

    const intVal = (id, def = null) => {
        const v = val(id, def);
        if (v === null || v === undefined || v === '') return def;
        const result = parseInt(v);
        if (isNaN(result)) {
            console.warn(`Invalid integer value for ${id}: '${v}', using default:`, def);
            return def;
        }
        return result;
    };

    const floatVal = (id, def = null) => {
        const v = val(id, def);
        if (v === null || v === undefined || v === '') return def;
        const result = parseFloat(v);
        if (isNaN(result)) {
            console.warn(`Invalid float value for ${id}: '${v}', using default:`, def);
            return def;
        }
        return result;
    };

    const membraneSeqInput = document.getElementById('membraneSeqInput');
    const membraneSeqValue = membraneSeqInput ? membraneSeqInput.value.trim() : '';
    const membraneSeqs = membraneSeqValue ? membraneSeqValue.split(',').map(s => s.trim()).filter(s => s) : [];

    let membraneRegions = [];
    const selectedStructures = state.getSelectedStructures();

    if (selectedStructures.length > 0) {
        let currentOffset = 0;

        selectedStructures.forEach(structure => {
            // 1. Collect manual regions for this sequence/chain
            let structSeqId = null;
            let currentSequence = null;
            if (structure.sequences && structure.sequences.length > 0) {
                structSeqId = `${structure.filename}_${structure.sequences[0].id}`;
                currentSequence = structure.sequences[0].sequence;
            }

            if (structSeqId && state.membraneRegions[structSeqId]) {
                const chainRegions = state.membraneRegions[structSeqId]
                    .filter(region =>
                        typeof region === 'object' &&
                        region !== null &&
                        typeof region.start === 'number' &&
                        typeof region.end === 'number'
                    )
                    .map(region => ({
                        start: Number(region.start) + currentOffset + 1, // 1-based
                        end: Number(region.end) + currentOffset + 1      // 1-based
                    }));

                membraneRegions = membraneRegions.concat(chainRegions);
            }

            // 2. Also search for membrane sequences in this specific chain
            if (currentSequence && membraneSeqs.length > 0) {
                const seqUpper = currentSequence.toUpperCase();
                membraneSeqs.forEach(q => {
                    const qUpper = q.toUpperCase();
                    if (!qUpper) return;

                    let startIdx = 0;
                    while (true) {
                        const k = seqUpper.indexOf(qUpper, startIdx);
                        if (k < 0) break;

                        const regStart = k + currentOffset + 1; // 1-based
                        const regEnd = k + qUpper.length + currentOffset; // 1-based

                        // Check if this region is already captured (avoid duplicates)
                        const exists = membraneRegions.some(r => r.start === regStart && r.end === regEnd);
                        if (!exists) {
                            membraneRegions.push({
                                start: regStart,
                                end: regEnd
                            });
                        }
                        startIdx = k + 1;
                    }
                });
            }

            // Increment offset by sequence length
            if (structure.sequences && structure.sequences.length > 0) {
                currentOffset += structure.sequences[0].length;
            }
        });

        console.log(`Collected ${membraneRegions.length} membrane regions from ${selectedStructures.length} chains`);
    } else if (sequenceId && state.membraneRegions[sequenceId]) {
        // Fallback for single sequence ID if no structures selected (legacy)
        membraneRegions = state.membraneRegions[sequenceId]
            .filter(region =>
                typeof region === 'object' &&
                region !== null &&
                typeof region.start === 'number' &&
                typeof region.end === 'number'
            )
            .map(region => ({
                start: Number(region.start) + 1,
                end: Number(region.end) + 1
            }));
    }

    const hasMembraneRegions = membraneRegions.length > 0;
    const hasMembraneSeqs = membraneSeqs.length > 0;
    // Enable if either regions or sequences are provided, AND checkbox is checked
    // The UI auto-checks the checkbox when sequences are entered, so this is usually true
    const enableMembrane = (hasMembraneRegions || hasMembraneSeqs) && !!val('membrane', false);

    // Check if any selected structure is a PDB (has structure file)
    const hasPdbStructures = selectedStructures.some(s => s.type === 'pdb' && s.file_path);
    const pdbProvided = hasPdbStructures;
    // Always respect user's ColabFold checkbox
    const runColab = !!val('runColabfold', true);

    return {
        sequence_id: sequenceId,
        run_colabfold: runColab,
        use_structure_plddt: !!val('useStructurePlddt', false),
        colabfold_args: val('colabfoldArgs', '--num-models 1'),
        low_spec: !!val('lowSpec', false),
        mem_frac: floatVal('memoryFraction', 0.5),
        gpu: intVal('gpuDevice'),
        model_type: val('modelType', 'auto') || null,
        plddt_rigid: floatVal('plddtThreshold', 70.0),
        min_rb_len: intVal('minRigidLength', 12),
        bead_res_per_bead: intVal('beadSize', 10),
        model_disordered_as_beads: !!val('modelDisorderedAsBeads', false),
        frames: intVal('numFrames', 100000),
        steps_per_frame: intVal('stepsPerFrame', 10),
        barrier_radius: floatVal('barrierRadius', 100.0),
        membrane: enableMembrane,
        membrane_regions: enableMembrane ? membraneRegions : [],
        membrane_seqs: enableMembrane ? membraneSeqs : [], // New field passed to backend
        membrane_weight: floatVal('membraneWeight', 10.0),
        reuse: !!val('reuse', true),
        measure: !!val('measure', false),
        measure_plot: !!val('measurePlot', true),
        // Collect FP sites if measurement is enabled
        sites: (!!val('measure', false)) ? (() => {
            const siteList = [];
            const fpSites = state.fpSites;

            // FP1 sites (donor)
            if (fpSites.donor.site1) siteList.push(`${fpSites.donor.site1.chainId}:${fpSites.donor.site1.position}`);
            if (fpSites.donor.site2) siteList.push(`${fpSites.donor.site2.chainId}:${fpSites.donor.site2.position}`);

            // FP2 sites (acceptor)
            if (fpSites.acceptor.site1) siteList.push(`${fpSites.acceptor.site1.chainId}:${fpSites.acceptor.site1.position}`);
            if (fpSites.acceptor.site2) siteList.push(`${fpSites.acceptor.site2.chainId}:${fpSites.acceptor.site2.position}`);

            return siteList;
        })() : [],
        // Use the first PDB structure from selected structures
        af_pdb: pdbProvided ? selectedStructures.find(s => s.type === 'pdb' && s.file_path)?.file_path : null,
    };
}

export function showJobStatus(jobId, resultsUrl = null) {
    // Declare jobStatusDiv in outer scope
    let jobStatusDiv = null;

    try {
        const noJobEl = document.getElementById(DOM.noJob);
        if (noJobEl && noJobEl.style) {
            noJobEl.style.display = 'none';
        }

        jobStatusDiv = document.getElementById(DOM.jobStatus);
        if (jobStatusDiv && jobStatusDiv.style) {
            jobStatusDiv.style.display = 'block';
        } else {
            console.error('jobStatus element not found');
            return;
        }
    } catch (error) {
        console.error('Error in showJobStatus:', error);
        throw error;
    }

    // Set job ID (or show message if no active job)
    const jobIdEl = document.getElementById(DOM.jobId);
    if (jobIdEl) {
        if (jobId) {
            jobIdEl.textContent = jobId;
        } else if (resultsUrl) {
            jobIdEl.textContent = 'Previous job (completed)';
        } else {
            jobIdEl.textContent = '-';
        }
    }

    // Update results link using the new container in HTML
    const resultsLinkContainer = document.getElementById('resultsLinkContainer');
    const resultsLink = document.getElementById('resultsLink');

    if (resultsUrl && resultsLinkContainer && resultsLink) {
        resultsLink.href = resultsUrl;
        resultsLink.target = '_blank';
        resultsLinkContainer.style.display = 'block';
    } else if (resultsLinkContainer) {
        resultsLinkContainer.style.display = 'none';
    }

    // If no jobId, hide status badge and cancel button (results-only mode)
    if (!jobId) {
        const statusBadge = document.getElementById('statusBadge');
        const cancelBtn = document.getElementById('cancelJobBtn');
        if (statusBadge && statusBadge.parentElement) {
            statusBadge.parentElement.style.display = 'none';
        }
        if (cancelBtn) {
            cancelBtn.style.display = 'none';
        }
    }
}

function fallbackCopyTextToClipboard(text, button) {
    const textArea = document.createElement("textarea");
    textArea.value = text;

    // Avoid scrolling to bottom
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";

    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        const successful = document.execCommand('copy');
        if (successful) {
            const originalHtml = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check"></i>';
            button.classList.add('text-success');
            setTimeout(() => {
                button.innerHTML = originalHtml;
                button.classList.remove('text-success');
            }, 2000);
        }
    } catch (err) {
        // Silently fail - clipboard API failures are expected on HTTP
    }

    document.body.removeChild(textArea);
}

export function startStatusPolling() {
    if (state.statusInterval) clearInterval(state.statusInterval);

    state.statusInterval = setInterval(async () => {
        if (!state.currentJobId) return;

        try {
            const data = await fetchStatus(state.currentJobId);

            updateJobStatus(data);

            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(state.statusInterval);
                const completedJobId = state.currentJobId; // Store ID before clearing
                state.currentJobId = null; // Clear from state

                // Save results URL for completed jobs
                if (data.results_url) {
                    state.lastJobResultsUrl = data.results_url;
                }

                // Save state to persist results URL
                const { saveState } = await import('./persistence.js');
                saveState();

                if (data.status === 'completed') {
                    loadResults(completedJobId); // Pass ID explicitly
                    showSuccess('Job completed successfully!');
                } else {
                    showError('Job failed: ' + (data.error || 'Unknown error'));
                }
            }
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 2000);
}

async function updateJobStatus(data) {
    const statusBadge = document.getElementById(DOM.statusBadge);
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressText = document.getElementById('progressText');
    const statusMessage = document.getElementById('statusMessage');

    // Update status badge
    if (statusBadge) {
        statusBadge.textContent = data.status.replace('_', ' ').toUpperCase();
        statusBadge.className = `badge bg-${getStatusColor(data.status)}`;
    }

    // Update progress bar with detailed progress from API
    const progressData = data.progress || {};
    const progress = progressData.progress_percent || getStatusProgress(data.status);

    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${progress}%`;
        if (progressPercent) progressPercent.textContent = progress;
        progressBar.className = `progress-bar progress-bar-striped progress-bar-animated bg-${getStatusColor(data.status)}`;
    }

    // Update status message
    if (statusMessage) {
        statusMessage.textContent = getStatusMessage(data.status, progressData);
    }

    // Show/hide cancel button based on status
    const cancelBtn = document.getElementById('cancelJobBtn');
    if (cancelBtn) {
        // Only show cancel for active jobs
        if (data.status === 'queued' || data.status === 'running' ||
            data.status === 'colabfold_complete' || data.status === 'sampling_complete') {
            cancelBtn.style.display = 'inline-block';
        } else {
            cancelBtn.style.display = 'none';
        }
    }

    // Fetch and display sampling progress if running/sampling
    if (data.status === 'running' || data.status === 'sampling_complete') {
        await updateSamplingProgress(data.job_id);
    } else {
        // Hide sampling progress for other statuses
        const samplingDiv = document.getElementById('samplingProgress');
        if (samplingDiv) samplingDiv.style.display = 'none';
    }

    // Update progress details section
    updateProgressDetails(data, progressData);

    // Update results link if available
    if (data.results_url) {
        const jobStatusDiv = document.getElementById(DOM.jobStatus);
        let resultsContainer = jobStatusDiv.querySelector('.results-container');
        if (!resultsContainer) {
            showJobStatus(data.job_id, data.results_url);
        }
    }
}

async function updateSamplingProgress(jobId) {
    try {
        const response = await fetch(`/api/job_sampling_log/${jobId}`);
        if (!response.ok) return;

        const data = await response.json();

        // Show sampling progress section
        const samplingDiv = document.getElementById('samplingProgress');
        const samplingBar = document.getElementById('samplingProgressBar');
        const samplingText = document.getElementById('samplingProgressText');
        const samplingBadge = document.getElementById('samplingFrameBadge');

        if (data.current_frame > 0 || data.total_frames > 0) {
            if (samplingDiv) samplingDiv.style.display = 'block';

            const samplingPercent = data.progress_percent || 0;

            if (samplingBar) {
                samplingBar.style.width = `${samplingPercent}%`;
            }
            if (samplingText) {
                samplingText.textContent = `${samplingPercent}%`;
            }
            if (samplingBadge) {
                samplingBadge.textContent = `Frame ${data.current_frame.toLocaleString()} / ${data.total_frames.toLocaleString()}`;
            }
        } else {
            if (samplingDiv) samplingDiv.style.display = 'none';
        }
    } catch (error) {
        console.log('Sampling progress not available:', error);
    }
}

function getStatusMessage(status, progressData) {
    switch (status) {
        case 'queued':
            if (progressData.queue_position) {
                return `Waiting in queue (position ${progressData.queue_position})`;
            }
            return 'Job queued, waiting to start...';
        case 'running':
            if (progressData.current_step) {
                return `Running: ${progressData.current_step.replace('_', ' ')}`;
            }
            return 'Job is running...';
        case 'colabfold_complete':
            return 'ColabFold prediction complete, preparing sampling...';
        case 'sampling_complete':
            return 'Sampling complete, analyzing results...';
        case 'completed':
            return 'Job completed successfully!';
        case 'failed':
            return 'Job failed. Check error details.';
        default:
            return 'Processing...';
    }
}

function updateProgressDetails(data, progressData) {
    // Find or create progress details container
    let detailsContainer = document.getElementById('progressDetails');
    if (!detailsContainer) {
        const jobStatusDiv = document.getElementById(DOM.jobStatus);
        detailsContainer = document.createElement('div');
        detailsContainer.id = 'progressDetails';
        detailsContainer.className = 'progress-details mt-3 p-3 bg-light rounded';
        jobStatusDiv.appendChild(detailsContainer);
    }

    let html = '';

    if (data.status === 'queued') {
        // Show queue position and estimated wait time
        if (progressData.queue_position) {
            html += `
                <div class="alert alert-info mb-2">
                    <strong><i class="fas fa-clock me-2"></i>Position in Queue: ${progressData.queue_position}</strong>
                    ${progressData.total_queued ? `of ${progressData.total_queued} jobs` : ''}
                    ${progressData.estimated_wait_formatted ? `<br><small>Estimated wait: ${progressData.estimated_wait_formatted}</small>` : ''}
                </div>
            `;
        }
    } else if (data.status === 'running' || data.status === 'colabfold_complete' || data.status === 'sampling_complete') {
        // Show current step and progress


        // Show CLI output
        if (progressData.cli_output && progressData.cli_output.length > 0) {
            html += `
                <div class="cli-output mt-2">
                    <strong><i class="fas fa-terminal me-2"></i>CLI Output:</strong>
                    <pre class="bg-dark text-light p-2 rounded mt-1" style="max-height: 200px; overflow-y: auto; font-size: 0.75rem;">${formatCliOutput(progressData.cli_output)}</pre>
                </div>
            `;
        }
    } else if (data.status === 'completed') {
        html += `
            <div class="alert alert-success mb-2">
                <strong><i class="fas fa-check-circle me-2"></i>Job Completed Successfully!</strong>
            </div>
        `;
    } else if (data.status === 'failed') {
        html += `
            <div class="alert alert-danger mb-2">
                <strong><i class="fas fa-exclamation-triangle me-2"></i>Job Failed</strong>
                ${data.error ? `<br><small>${data.error}</small>` : ''}
            </div>
        `;
    }

    detailsContainer.innerHTML = html;
}

function formatCliOutput(output) {
    return output.map(line => {
        if (typeof line === 'string') return line;
        const timestamp = line.timestamp || '';
        const level = line.level || 'INFO';
        const message = line.message || '';
        const color = level === 'ERROR' ? '#ff6b6b' : level === 'WARNING' ? '#ffd93d' : '#6c757d';
        return `[${timestamp}] <span style="color: ${color}">${level}</span> ${message}`;
    }).join('\n');
}

function getStatusColor(status) {
    const colors = {
        'queued': 'warning',
        'running': 'info',
        'colabfold_complete': 'primary',
        'sampling_complete': 'secondary',
        'completed': 'success',
        'failed': 'danger'
    };
    return colors[status] || 'secondary';
}

function getStatusProgress(status) {
    const progress = {
        'queued': 10,
        'running': 25,
        'colabfold_complete': 50,
        'sampling_complete': 80,
        'completed': 100,
        'failed': 100
    };
    return progress[status] || 0;
}

export async function cancelJob() {
    if (!state.currentJobId) {
        showError('No active job to cancel');
        return;
    }

    if (!confirm('Are you sure you want to cancel this job?')) {
        return;
    }

    try {
        const response = await fetch(`/api/job_cancel/${state.currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            const data = await response.json();
            showSuccess('Job cancelled successfully');

            // Clear job state
            state.currentJobId = null;
            state.lastJobResultsUrl = null;

            // Save cleared state
            const { saveState } = await import('./persistence.js');
            saveState();

            // Stop polling and hide cancel button
            if (state.statusInterval) {
                clearInterval(state.statusInterval);
            }

            const cancelBtn = document.getElementById('cancelJobBtn');
            if (cancelBtn) cancelBtn.style.display = 'none';

            // Refresh status display
            await refreshJobStatus();
        } else {
            const error = await response.json();
            showError('Failed to cancel job: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Cancel job error:', error);
        showError('Failed to cancel job: ' + error.message);
    }
}

async function loadResults(jobId) {
    try {
        const idToUse = jobId || state.currentJobId;
        if (!idToUse) {
            console.error('No job ID available for loading results');
            return;
        }
        const data = await fetchResults(idToUse);
        displayResults(data);
    } catch (error) {
        console.error('Failed to load results:', error);
        showError('Failed to load results. Please check console.');
    }
}

// Expose functions globally for HTML onclick handlers
window.cancelJob = cancelJob;
window.refreshJobStatus = refreshJobStatus;

function displayResults(data) {
    const resultsSection = document.getElementById(DOM.resultsSection);
    const downloadLinks = document.getElementById(DOM.downloadLinks);

    downloadLinks.innerHTML = '';

    if (data.files && Array.isArray(data.files) && data.files.length > 0) {
        data.files.forEach(file => {
            if (file.type === 'rmf') {
                const link = document.createElement('a');
                // URL-encode the file path to handle subdirectories correctly
                const encodedPath = encodeURIComponent(file.path);
                const jobId = data.job_id || state.currentJobId;
                link.href = `/api/download/${jobId}/${encodedPath}`;
                link.className = 'btn btn-sm btn-outline-primary me-2 mb-2';
                link.innerHTML = `<i class="fas fa-download me-1"></i>${file.name}`;
                link.download = file.name; // Set download attribute to suggest filename
                downloadLinks.appendChild(link);
            }
        });
    } else {
        // Optionally show a message if no files are found, but keeping it empty is also fine
        // or verify if data.error exists
        if (data.error) {
            showError('Error loading results: ' + data.error);
        } else if (!data.files || data.files.length === 0) {
            // Maybe show a placeholder
        }
    }

    resultsSection.style.display = 'block';
}

function refreshJobStatus() {
    if (state.currentJobId) {
        pollJobStatus(state.currentJobId);
    }
}
