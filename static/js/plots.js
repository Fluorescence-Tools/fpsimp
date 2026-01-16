/**
 * Plots module for pLDDT visualization and interaction
 */

let plddtChart = null;
let currentSegmentsByChain = null;

// Color constants
const COLORS = {
    plddt: '#0053D6',
    plddtFill: 'rgba(0, 83, 214, 0.1)',
    rigid: 'rgba(200, 200, 200, 0.2)',
    rigidBorder: 'rgba(180, 180, 180, 0.5)',
    linker: 'rgba(255, 165, 0, 0.2)', // Orange for linker
    linkerBorder: 'rgba(255, 140, 0, 0.5)',
    fp: 'rgba(0, 255, 0, 0.2)',
    fpBorder: 'rgba(0, 255, 0, 0.6)',
    membrane: 'rgba(255, 0, 0, 0.2)',
    membraneBorder: 'rgba(255, 0, 0, 0.6)'
};

/**
 * Render the pLDDT chart
 * @param {string} canvasId - DOM ID of the canvas
 * @param {Object} plddtByChain - Dict {chainId: {residueIdx: score}}
 * @param {Object} segmentsByChain - Dict {chainId: [segmentObjects]}
 * @param {Object} options - { interactive: boolean, onSegmentChange: callback }
 */
export function renderPlddtChart(canvasId, plddtByChain, segmentsByChain, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        console.error(`Canvas element with ID '${canvasId}' not found.`);
        return;
    }

    // Destroy existing chart if it exists
    if (plddtChart) {
        plddtChart.destroy();
    }

    // Store current segments for mutation during interaction
    // Deep copy to avoid mutating the passed object directly until callback
    currentSegmentsByChain = JSON.parse(JSON.stringify(segmentsByChain));

    // 1. Prepare Data for Chart.js
    // We want a continuous sequence of points, but labeled with Chain ID
    const flattenedLabels = [];
    const flattenedScores = [];
    const chainBoundaries = []; // To draw vertical separators between chains
    let globalIndex = 0;

    // Sort chains to ensure deterministic order (e.g. A, B, C)
    const sortedChains = Object.keys(plddtByChain).sort();

    // Map to convert global index back to (chain, residue)
    const indexToResidue = [];

    sortedChains.forEach((chainId, chainIdx) => {
        const plddtData = plddtByChain[chainId];
        if (!plddtData) {
            console.warn(`No pLDDT data for chain ${chainId}`);
            return;
        }
        const residues = Object.keys(plddtData).map(Number).sort((a, b) => a - b);

        residues.forEach(resNum => {
            flattenedLabels.push(`${chainId}:${resNum}`);
            flattenedScores.push(plddtData[resNum]);
            indexToResidue.push({ chainId, resNum, globalIndex });
            globalIndex++;
        });

        // Mark end of chain
        if (chainIdx < sortedChains.length - 1) {
            chainBoundaries.push(globalIndex - 0.5); // Between last of this and first of next
        }
    });

    // 2. Prepare Annotations
    const annotations = {};

    // Helper to find global x-index range for a segment
    // Segment start/end are 0-based relative to the chain sequence?
    // Let's verify standard: usually backend uses 0-based for start, end.
    // Our backend `segments_from_plddt` returns start/end as indices. 
    // We need to map these chain-local 0-based indices to our global linear index.

    // Build offset map
    const chainOffsets = {};
    let currentOffset = 0;
    sortedChains.forEach(chainId => {
        chainOffsets[chainId] = currentOffset;
        const len = Object.keys(plddtByChain[chainId]).length;
        currentOffset += len;
    });

    // Add Segment Annotations (Rectangles background)
    sortedChains.forEach(chainId => {
        const chainSegments = currentSegmentsByChain[chainId] || [];
        const offset = chainOffsets[chainId];

        chainSegments.forEach((seg, idx) => {
            const id = `${chainId}_seg_${idx}`;
            // start/end are 1-based inclusive indices from backend
            // Convert to 0-based for global coordinate mapping
            const xMin = offset + (seg.start - 1);
            const xMax = offset + (seg.end - 1);

            annotations[id] = {
                type: 'box',
                xMin: xMin - 0.5, // Align with pixel grid roughly
                xMax: xMax + 0.5,
                yMin: 0,
                yMax: 100,
                backgroundColor: seg.kind === 'core' ? COLORS.rigid : (seg.kind === 'fp' ? COLORS.fp : COLORS.linker),
                borderColor: 'transparent',
                borderWidth: 0,
                label: {
                    display: true,
                    content: seg.kind === 'core' ? 'Rigid' : (seg.kind === 'fp' ? (seg.name || 'FP') : 'Linker'),
                    position: seg.kind === 'linker' ? 'end' : 'start',
                    font: { size: 10 }
                },
                // Custom data to identify this segment in events
                chain_id: chainId,
                segment_index: idx
            };
        });
    });

    // Add Boundary Lines (Visual only - controlled via Table)
    if (options.interactive) {
        sortedChains.forEach(chainId => {
            const chainSegments = currentSegmentsByChain[chainId] || [];
            const offset = chainOffsets[chainId];

            for (let i = 0; i < chainSegments.length - 1; i++) {
                const seg1 = chainSegments[i];
                const boundaryResIdx = seg1.end; // 1-based index
                // Boundary visual position
                const globalX = offset + (boundaryResIdx - 1) + 0.5;

                const boundaryId = `bound_${chainId}_${i}`;

                annotations[boundaryId] = {
                    type: 'line',
                    mode: 'vertical',
                    scaleID: 'x',
                    value: globalX,
                    borderColor: 'rgba(0,0,0,0.5)',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    z: 50
                    // Label removed as requested (no handles)
                };
            }
        });
    }

    // Chart Configuration
    const config = {
        type: 'line',
        data: {
            // No 'labels' key (uses linear x)
            datasets: [{
                label: 'pLDDT Score',
                // Data as {x, y} points
                data: flattenedScores.map((score, i) => ({ x: i, y: score })),
                borderColor: COLORS.plddt,
                backgroundColor: COLORS.plddtFill,
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            animation: false, // Disable animation to prevent "pop-up" effect on update
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'nearest',
                intersect: false,
                axis: 'x'
            },
            plugins: {
                title: { display: true, text: 'pLDDT Score and Segmentation' },
                legend: { display: false },
                annotation: {
                    annotations: annotations,
                    clip: false
                },
                tooltip: {
                    callbacks: {
                        title: function (context) {
                            // Map linear X back to label
                            const idx = Math.round(context[0].parsed.x);
                            return flattenedLabels[idx] || '';
                        },
                        label: function (context) {
                            return `pLDDT: ${context.parsed.y}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', // Explicit linear
                    title: { display: true, text: 'Residue (Chain:Number)' },
                    ticks: {
                        stepSize: 1, // Optional, might be too dense
                        maxTicksLimit: 20,
                        callback: function (val, index) {
                            // Map integer value back to label string
                            const idx = Math.round(val);
                            if (idx >= 0 && idx < flattenedLabels.length) {
                                return flattenedLabels[idx];
                            }
                            return '';
                        }
                    },
                    grid: { display: false },
                    min: 0,
                    max: globalIndex - 1
                },
                y: {
                    title: { display: true, text: 'Score' },
                    min: 0,
                    max: 100,
                    ticks: { stepSize: 20 }
                }
            }
        }
    };

    plddtChart = new Chart(ctx, config);
}

// Export functions
window.plots = {
    renderPlddtChart: renderPlddtChart
};
