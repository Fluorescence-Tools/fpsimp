/**
 * Results page functionality for displaying intermediate and final results
 */

// Chart.js for pLDDT visualization
let plddtChart = null;

/**
 * Fetch intermediate results for a job
 */
async function fetchIntermediateResults(jobId) {
    try {
        const response = await fetch(`/api/job_intermediate/${jobId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching intermediate results:', error);
        return null;
    }
}

/**
 * Fetch and display diagnostic information
 */
async function displayDiagnostics(jobId) {
    try {
        const response = await fetch(`/api/job_diagnostics/${jobId}`);
        if (!response.ok) return;

        const diag = await response.json();
        const container = document.getElementById('diagnosticContent');
        if (!container) return;

        let html = '<div class="row g-3">';

        // Queue Information
        if (diag.queue_info) {
            html += '<div class="col-md-6">';
            html += '<h6><i class="fas fa-list me-1"></i>Queue Status</h6>';
            html += '<ul class="mb-0">';
            html += `<li><strong>Jobs Queued:</strong> ${diag.queue_info.queued_count || 0}</li>`;
            html += `<li><strong>Jobs Running:</strong> ${diag.queue_info.running_count || 0}</li>`;

            if (diag.queue_info.job_position) {
                html += `<li><strong>Your Position:</strong> ${diag.queue_info.job_position}</li>`;
                if (diag.queue_info.estimated_wait_minutes) {
                    html += `<li><strong>Est. Wait:</strong> ~${diag.queue_info.estimated_wait_minutes} minutes</li>`;
                }
            }
            html += '</ul>';
            html += '</div>';
        }

        // Worker Information
        if (diag.celery_info) {
            html += '<div class="col-md-6">';
            html += '<h6><i class="fas fa-server me-1"></i>Worker Status</h6>';
            html += '<ul class="mb-0">';

            const workersOnline = diag.celery_info.workers_online || 0;
            const statusIcon = workersOnline > 0 ? '<i class="fas fa-check-circle text-success"></i>' : '<i class="fas fa-times-circle text-danger"></i>';
            html += `<li><strong>Workers Online:</strong> ${statusIcon} ${workersOnline}</li>`;

            if (diag.celery_info.worker_names && diag.celery_info.worker_names.length > 0) {
                html += `<li><strong>Workers:</strong> ${diag.celery_info.worker_names.join(', ')}</li>`;
            }

            if (diag.celery_info.active_tasks_count !== undefined) {
                html += `<li><strong>Active Tasks:</strong> ${diag.celery_info.active_tasks_count}</li>`;
            }

            if (diag.celery_info.celery_queue_length !== undefined) {
                html += `<li><strong>Celery Queue Length:</strong> ${diag.celery_info.celery_queue_length}</li>`;
            }

            if (workersOnline === 0) {
                html += '<li class="text-danger"><strong>⚠️ No workers online!</strong> Job will not process until a worker starts.</li>';
            }

            html += '</ul>';
            html += '</div>';
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Error fetching diagnostics:', error);
        const container = document.getElementById('diagnosticContent');
        if (container) {
            container.innerHTML = '<p class="text-muted mb-0">Unable to load diagnostic information.</p>';
        }
    }
}

/**
 * Display sequence with pLDDT coloring
 */
function displaySequenceWithPlddt(container, sequence, plddt, fpDomains, membraneRegions) {
    if (!sequence || !plddt) return;

    container.innerHTML = '';

    // Create sequence display
    const seqDiv = document.createElement('div');
    seqDiv.className = 'sequence-display-results';
    seqDiv.style.fontFamily = 'monospace';
    seqDiv.style.fontSize = '12px';
    seqDiv.style.lineHeight = '1.8';
    seqDiv.style.wordBreak = 'break-all';
    seqDiv.style.whiteSpace = 'pre-wrap';

    for (let i = 0; i < sequence.length; i++) {
        const span = document.createElement('span');
        span.textContent = sequence[i];
        span.style.padding = '2px 1px';
        span.style.cursor = 'pointer';
        span.title = `Position ${i + 1}: ${sequence[i]}, pLDDT: ${plddt[i] ? plddt[i].toFixed(1) : 'N/A'}`;

        // Color by pLDDT
        if (plddt[i] !== undefined) {
            const score = plddt[i];
            if (score >= 90) {
                span.style.backgroundColor = '#0053D6'; // Dark blue
                span.style.color = 'white';
            } else if (score >= 70) {
                span.style.backgroundColor = '#65CBF3'; // Light blue
            } else if (score >= 50) {
                span.style.backgroundColor = '#FFDB13'; // Yellow
            } else {
                span.style.backgroundColor = '#FF7D45'; // Orange
            }
        }

        // Mark FP domains with border
        if (fpDomains) {
            for (const fp of fpDomains) {
                if (i >= fp.start && i <= fp.end) {
                    span.style.border = '2px solid #00FF00';
                    span.title += ` | FP: ${fp.name}`;
                }
            }
        }

        // Mark Membrane regions with border
        if (membraneRegions) {
            for (const mem of membraneRegions) {
                if (i >= mem.start && i <= mem.end) {
                    span.style.border = '2px solid #FF0000'; // Red border for membrane
                    span.title += ` | Membrane Region`;
                }
            }
        }

        seqDiv.appendChild(span);

        // Add line breaks every 80 characters
        if ((i + 1) % 80 === 0) {
            seqDiv.appendChild(document.createElement('br'));
        }
    }

    container.appendChild(seqDiv);
}

/**
 * Display pLDDT chart with segmentation, FPs, and membrane regions
 */
function displayPlddtChart(canvasId, plddt, segments, fpDomains, membraneRegions) {
    if (!plddt || plddt.length === 0) return;

    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing chart
    if (plddtChart) {
        plddtChart.destroy();
    }

    const positions = Object.keys(plddt).map(Number);
    const scores = positions.map(pos => plddt[pos]);

    // Create annotation boxes for segmentation
    const annotations = {};

    // Add segmentation regions
    if (segments && segments.length > 0) {
        segments.forEach((seg, idx) => {
            annotations[`seg_${idx}`] = {
                type: 'box',
                xMin: seg.start,
                xMax: seg.end,
                yMin: 0,
                yMax: 100,
                backgroundColor: seg.kind === 'core'
                    ? 'rgba(200, 200, 200, 0.2)'
                    : 'rgba(100, 100, 100, 0.3)',
                borderColor: seg.kind === 'core'
                    ? 'rgba(180, 180, 180, 0.5)'
                    : 'rgba(80, 80, 80, 0.5)',
                borderWidth: 1,
                label: {
                    display: false,
                    content: seg.kind === 'core' ? 'Rigid' : 'Flexible'
                }
            };
        });
    }

    // Add FP domain regions
    if (fpDomains && fpDomains.length > 0) {
        fpDomains.forEach((fp, idx) => {
            annotations[`fp_${idx}`] = {
                type: 'box',
                xMin: fp.start,
                xMax: fp.end,
                yMin: 85,
                yMax: 100,
                backgroundColor: fp.color === 'magenta' ? 'rgba(255, 0, 255, 0.2)' :
                    (fp.color === 'cyan' ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 255, 0, 0.2)'),
                borderColor: fp.color === 'magenta' ? 'rgba(255, 0, 255, 0.6)' :
                    (fp.color === 'cyan' ? 'rgba(0, 255, 255, 0.6)' : 'rgba(0, 255, 0, 0.6)'),
                borderWidth: 2,
                label: {
                    display: true,
                    content: fp.name,
                    position: 'start',
                    color: '#000',
                    font: {
                        size: 10,
                        weight: 'bold'
                    }
                }
            };
        });
    }

    // Add membrane regions
    if (membraneRegions && membraneRegions.length > 0) {
        membraneRegions.forEach((mem, idx) => {
            annotations[`mem_${idx}`] = {
                type: 'box',
                xMin: mem.start,
                xMax: mem.end,
                yMin: 0,
                yMax: 15,
                backgroundColor: 'rgba(255, 0, 0, 0.2)',
                borderColor: 'rgba(255, 0, 0, 0.6)',
                borderWidth: 2,
                label: {
                    display: true,
                    content: 'Membrane',
                    position: 'start',
                    color: '#000',
                    font: {
                        size: 10,
                        weight: 'bold'
                    }
                }
            };
        });
    }

    plddtChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: positions,
            datasets: [{
                label: 'pLDDT Score',
                data: scores,
                borderColor: '#0053D6',
                backgroundColor: 'rgba(0, 83, 214, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'pLDDT with Segmentation, FPs, and Membrane Regions'
                },
                legend: {
                    display: true,
                    labels: {
                        generateLabels: function (chart) {
                            return [
                                {
                                    text: 'pLDDT Score',
                                    fillStyle: '#0053D6',
                                    strokeStyle: '#0053D6',
                                    lineWidth: 2
                                },
                                {
                                    text: 'Rigid Body',
                                    fillStyle: 'rgba(200, 200, 200, 0.3)',
                                    strokeStyle: 'rgba(180, 180, 180, 0.5)',
                                    lineWidth: 1
                                },
                                {
                                    text: 'Flexible Linker',
                                    fillStyle: 'rgba(100, 100, 100, 0.4)',
                                    strokeStyle: 'rgba(80, 80, 80, 0.5)',
                                    lineWidth: 1
                                },
                                {
                                    text: 'Fluorescent Protein',
                                    fillStyle: 'rgba(0, 255, 0, 0.3)',
                                    strokeStyle: 'rgba(0, 255, 0, 0.6)',
                                    lineWidth: 2
                                },
                                {
                                    text: 'Membrane Region',
                                    fillStyle: 'rgba(255, 0, 0, 0.3)',
                                    strokeStyle: 'rgba(255, 0, 0, 0.6)',
                                    lineWidth: 2
                                }
                            ];
                        }
                    }
                },
                annotation: {
                    annotations: annotations
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Residue Position'
                    },
                    ticks: {
                        maxTicksLimit: 20
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'pLDDT Score'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

/**
 * Display segmentation information
 */
function displaySegmentation(container, segments) {
    if (!segments || segments.length === 0) {
        container.innerHTML = '<p class="text-muted">No segmentation data available</p>';
        return;
    }

    container.innerHTML = '';

    const table = document.createElement('table');
    table.className = 'table table-sm table-bordered';
    table.innerHTML = `
        <thead>
            <tr>
                <th>Type</th>
                <th>Start</th>
                <th>End</th>
                <th>Length</th>
                <th>Name</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');
    segments.forEach(seg => {
        const row = document.createElement('tr');
        const typeClass = seg.kind === 'core' ? 'bg-secondary bg-opacity-10' : 'bg-dark bg-opacity-25';
        row.className = typeClass;
        row.innerHTML = `
            <td><strong>${seg.kind}</strong></td>
            <td>${seg.start + 1}</td>
            <td>${seg.end + 1}</td>
            <td>${seg.end - seg.start + 1}</td>
            <td>${seg.name || '-'}</td>
        `;
        tbody.appendChild(row);
    });

    container.appendChild(table);
}

/**
 * Display FP domains
 */
function displayFpDomains(container, fpDomains) {
    if (!fpDomains || fpDomains.length === 0) {
        container.innerHTML = '<p class="text-muted">No fluorescent proteins detected</p>';
        return;
    }

    container.innerHTML = '';

    const table = document.createElement('table');
    table.className = 'table table-sm table-bordered';
    table.innerHTML = `
        <thead>
            <tr>
                <th>FP Name</th>
                <th>Start</th>
                <th>End</th>
                <th>Length</th>
                <th>Identity</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');
    fpDomains.forEach(fp => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${fp.name}</strong></td>
            <td>${fp.start + 1}</td>
            <td>${fp.end + 1}</td>
            <td>${fp.end - fp.start + 1}</td>
            <td>${(fp.identity * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });

    container.appendChild(table);
}

/**
 * Display PMI topology file as table
 */
function displayTopology(container, topologyLines) {
    if (!topologyLines || topologyLines.length === 0) {
        container.innerHTML = '<p class="text-muted">No topology data available</p>';
        return;
    }

    container.innerHTML = '';

    const pre = document.createElement('pre');
    pre.style.fontSize = '11px';
    pre.style.maxHeight = '400px';
    pre.style.overflow = 'auto';
    pre.style.backgroundColor = '#f8f9fa';
    pre.style.padding = '1rem';
    pre.style.borderRadius = '0.25rem';
    pre.textContent = topologyLines.join('\n');

    container.appendChild(pre);
}

/**
 * Fetch and display IMP sampling log
 */
async function displaySamplingLog(container, jobId) {
    try {
        const response = await fetch(`/api/job_sampling_log/${jobId}`);
        if (!response.ok) throw new Error('Log not available');

        const data = await response.json();
        if (data.log) {
            // Check if structure already exists
            let progressDiv = container.querySelector('.progress-container');
            let logSection = container.querySelector('.log-section');

            // 1. Handle Progress Bar
            const jobStatus = document.getElementById('jobStatus')?.dataset?.status;
            const isTerminal = jobStatus === 'completed' || jobStatus === 'failed';

            if ((data.current_frame > 0 || data.total_frames > 0) && !isTerminal) {
                if (!progressDiv) {
                    // Create if doesn't exist
                    progressDiv = document.createElement('div');
                    progressDiv.className = 'mb-3 progress-container';
                    // We prepend or append depending on if log section exists, 
                    // but typically we want it at the top.
                    if (logSection) {
                        container.insertBefore(progressDiv, logSection);
                    } else {
                        container.appendChild(progressDiv);
                    }
                }

                // Update content
                progressDiv.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span><strong>Frame ${data.current_frame.toLocaleString()} / ${data.total_frames.toLocaleString()}</strong></span>
                        <span class="badge bg-info">${data.progress_percent}%</span>
                    </div>
                    <div class="progress" style="height: 25px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" 
                             role="progressbar" 
                             style="width: ${data.progress_percent}%"
                             aria-valuenow="${data.progress_percent}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                            ${data.progress_percent}%
                        </div>
                    </div>
                `;
            } else if (progressDiv) {
                // Hide progress bar for terminal states
                progressDiv.style.display = 'none';
            }

            // 2. Handle Detailed Log Output
            if (!logSection) {
                // Create structure only once
                logSection = document.createElement('div');
                logSection.className = 'mt-3 log-section';
                logSection.innerHTML = `
                    <button class="btn btn-sm btn-outline-secondary mb-2" type="button" data-bs-toggle="collapse" data-bs-target="#detailedLogOutput" aria-expanded="false">
                        <i class="fas fa-terminal me-1"></i>Show/Hide Detailed Output
                    </button>
                    <div class="collapse" id="detailedLogOutput">
                        <pre style="font-size: 11px; background-color: #000; color: #0f0; padding: 1rem; border-radius: 0.25rem; white-space: pre-wrap; word-break: break-all;"></pre>
                    </div>
                `;
                container.appendChild(logSection);

                // Auto-scroll logic (only attach once)
                const collapseEl = logSection.querySelector('#detailedLogOutput');
                collapseEl.addEventListener('shown.bs.collapse', function () {
                    // When opened, we might want to scroll to bottom if desirable, 
                    // but user asked for "no rollover", so simply showing it is enough.
                    // If we want to scroll the WINDOW to see the new content, that's different.
                    // For now, removing the internal scroll behavior.
                });
            }

            // Update the log text
            const pre = logSection.querySelector('pre');
            if (pre) {
                // Only update if content changed to avoid cursor jumping if user was selecting text
                if (pre.textContent !== data.log) {
                    pre.textContent = data.log;
                }
            }

        } else {
            // Only show message if container is empty
            if (container.innerHTML.trim() === '') {
                container.innerHTML = '<p class="text-muted">No sampling log available yet</p>';
            }
        }
    } catch (error) {
        if (container.innerHTML.trim() === '') {
            container.innerHTML = '<p class="text-muted">Sampling log not available</p>';
        }
    }
}

/**
 * Initialize intermediate results display
 */
async function initializeIntermediateResults(jobId) {
    // Check if job was already completed (force reload to show final results)
    const initialStatus = document.getElementById('jobStatus')?.dataset?.status;
    if (initialStatus === 'completed' || initialStatus === 'failed') {
        // Page should have shown final results, might be stale - force one reload
        const lastReloadTime = sessionStorage.getItem('lastReload_' + jobId);
        const now = Date.now();
        if (!lastReloadTime || now - parseInt(lastReloadTime) > 5000) {
            sessionStorage.setItem('lastReload_' + jobId, now.toString());
            window.location.reload();
            return;
        }
    }

    // Display diagnostics if diagnostic section exists
    const diagnosticContainer = document.getElementById('diagnosticInfo');
    if (diagnosticContainer) {
        await displayDiagnostics(jobId);

        // Refresh diagnostics periodically for queued/running jobs
        setInterval(() => displayDiagnostics(jobId), 10000); // Every 10 seconds
    }

    const data = await fetchIntermediateResults(jobId);
    if (!data) return;

    // Display sequence with pLDDT
    const seqContainer = document.getElementById('sequenceContainer');
    if (seqContainer && data.sequence && data.plddt) {
        displaySequenceWithPlddt(seqContainer, data.sequence, data.plddt, data.fp_domains, data.membrane_regions);
    }

    // Display pLDDT chart with overlays
    if (data.plddt) {
        displayPlddtChart('plddtChart', data.plddt, data.segments, data.fp_domains, data.membrane_regions);
    }

    // Display segmentation
    const segContainer = document.getElementById('segmentationContainer');
    if (segContainer && data.segments) {
        displaySegmentation(segContainer, data.segments);
    }

    // Display FP domains
    const fpContainer = document.getElementById('fpDomainsContainer');
    if (fpContainer && data.fp_domains) {
        displayFpDomains(fpContainer, data.fp_domains);
    }

    // Display topology
    const topContainer = document.getElementById('topologyContainer');
    if (topContainer && data.topology) {
        displayTopology(topContainer, data.topology);
    }

    // Display sampling log
    const logContainer = document.getElementById('samplingLogContainer');
    if (logContainer) {
        displaySamplingLog(logContainer, jobId);

        // Refresh log every 5 seconds if job is running
        const jobStatus = document.getElementById('jobStatus')?.dataset?.status;
        if (jobStatus === 'running' || jobStatus === 'sampling_complete' || jobStatus === 'colabfold_complete') {
            setInterval(() => displaySamplingLog(logContainer, jobId), 5000);
        }
    }
}

/**
 * Refresh job status and update display dynamically
 */
async function refreshJobStatus(jobId) {
    try {
        const response = await fetch(`/api/job_status/${jobId}`);
        if (!response.ok) return;

        const statusData = await response.json();

        // Update status badge
        const statusBadge = document.querySelector('.status-badge');
        if (statusBadge && statusData.status) {
            statusBadge.textContent = statusData.status.replace('_', ' ').split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

            // Update badge color
            statusBadge.className = 'badge rounded-pill status-badge';
            if (statusData.status === 'completed') {
                statusBadge.classList.add('bg-success');
            } else if (statusData.status === 'failed') {
                statusBadge.classList.add('bg-danger');
            } else if (statusData.status === 'running' || statusData.status === 'sampling_complete' || statusData.status === 'colabfold_complete') {
                statusBadge.classList.add('bg-info');
            } else {
                statusBadge.classList.add('bg-secondary');
            }
        }

        // Update hidden status element
        const jobStatusElement = document.getElementById('jobStatus');
        if (jobStatusElement) {
            jobStatusElement.dataset.status = statusData.status;
        }

        // If job is completed or failed, reload page once to show final results
        if (statusData.status === 'completed' || statusData.status === 'failed') {
            window.location.reload();
            return;
        }

        // Refresh intermediate results if job is running
        if (statusData.status === 'running' || statusData.status === 'sampling_complete' || statusData.status === 'colabfold_complete') {
            await refreshIntermediateResults(jobId);
        }

    } catch (error) {
        console.error('Error refreshing job status:', error);
    }
}

/**
 * Refresh intermediate results without redrawing everything
 */
async function refreshIntermediateResults(jobId) {
    const data = await fetchIntermediateResults(jobId);
    if (!data) return;

    // Only update sampling log (most frequently changing)
    const logContainer = document.getElementById('samplingLogContainer');
    if (logContainer) {
        const currentScroll = logContainer.querySelector('pre')?.scrollTop;
        await displaySamplingLog(logContainer, jobId);

        // Restore scroll position if user hasn't scrolled to bottom
        const pre = logContainer.querySelector('pre');
        if (pre && currentScroll !== undefined && currentScroll < pre.scrollHeight - pre.clientHeight - 50) {
            pre.scrollTop = currentScroll;
        }
    }

    // Update segments/FPs only if chart hasn't been drawn yet
    if (!plddtChart && data.plddt) {
        displayPlddtChart('plddtChart', data.plddt, data.segments, data.fp_domains, data.membrane_regions);

        const seqContainer = document.getElementById('sequenceContainer');
        if (seqContainer && data.sequence) {
            displaySequenceWithPlddt(seqContainer, data.sequence, data.plddt, data.fp_domains, data.membrane_regions);
        }

        const segContainer = document.getElementById('segmentationContainer');
        if (segContainer && data.segments) {
            displaySegmentation(segContainer, data.segments);
        }

        const fpContainer = document.getElementById('fpDomainsContainer');
        if (fpContainer && data.fp_domains) {
            displayFpDomains(fpContainer, data.fp_domains);
        }

        const topContainer = document.getElementById('topologyContainer');
        if (topContainer && data.topology) {
            displayTopology(topContainer, data.topology);
        }
    }
}

/**
 * Start periodic status checks
 */
function startStatusPolling(jobId, intervalMs = 5000) {
    const jobStatusElement = document.getElementById('jobStatus');
    const currentStatus = jobStatusElement?.dataset?.status;

    // Don't poll if already completed or failed
    if (currentStatus === 'completed' || currentStatus === 'failed') {
        return;
    }

    // Initial refresh
    refreshJobStatus(jobId);

    // Set up interval
    const intervalId = setInterval(async () => {
        const statusElement = document.getElementById('jobStatus');
        const status = statusElement?.dataset?.status;

        // Stop polling if completed or failed
        if (status === 'completed' || status === 'failed') {
            clearInterval(intervalId);
            return;
        }

        await refreshJobStatus(jobId);
    }, intervalMs);

    // Store interval ID for cleanup
    window.statusPollingInterval = intervalId;
}

// Export for global use
window.initializeIntermediateResults = initializeIntermediateResults;
window.startStatusPolling = startStatusPolling;
window.refreshJobStatus = refreshJobStatus;
