
import { renderPlddtChart } from './plots.js';
import { renderTopologyTable } from './ui.js';
import { showStructurePreview } from './viewer.js';

/**
 * Visualizes the structure by showing the 3D viewer, pLDDT chart, and topology table.
 * @param {Object} structure - The structure object containing file_url, plddt, segments, etc.
 * @returns {Promise<void>}
 */
export async function visualizeStructure(structure) {
    if (!structure || !structure.file_url) {
        console.warn('visualizeStructure: No structure or file_url provided');
        return;
    }

    try {
        console.log('Visualizing structure object:', structure);
        if (structure.plddt) {
            console.log('pLDDT keys:', Object.keys(structure.plddt));
            console.log('pLDDT sample:', Object.values(structure.plddt)[0]);
        } else {
            console.log('Structure object has no plddt property (may be raw structure)');
        }
        if (structure.segments) {
            console.log('Segments keys:', Object.keys(structure.segments));
        } else {
            console.log('Structure object has no segments property (may be raw structure)');
        }

        // 1. Show 3D Viewer
        await showStructurePreview(
            structure.file_url,
            structure.filename,
            structure.sequences,
            structure.segments
        );

        // 2. Show Segmentation Section (pLDDT plot + table)
        const segmentationSection = document.getElementById('segmentationSection');
        if (segmentationSection) {
            segmentationSection.style.display = 'block';
        }

        // Check if we have valid pLDDT data
        const hasPlddt = structure.plddt && Object.keys(structure.plddt).length > 0;

        if (hasPlddt) {
            // 3. Render pLDDT Chart
            renderPlddtChart(
                'plddtChartLanding',
                structure.plddt,
                structure.segments,
                {
                    interactive: true,
                    onSegmentChange: (chainId, newSegments) => {
                        console.log(`Segment change detected for chain ${chainId}`, newSegments);

                        // Update structure object reference (updates in-memory state)
                        if (structure.segments) {
                            structure.segments[chainId] = newSegments;
                        }

                        // Re-render topology table
                        renderTopologyTable('topologyTableContainer', structure.segments);

                        // Re-render 3D viewer to reflect changes
                        showStructurePreview(
                            structure.file_url,
                            structure.filename,
                            structure.sequences,
                            structure.segments
                        );
                    }
                }
            );
        } else {
            // Show message that pLDDT is not available
            const chartCanvas = document.getElementById('plddtChartLanding');
            if (chartCanvas) {
                const ctx = chartCanvas.getContext('2d');
                ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
                ctx.font = '14px Arial';
                ctx.fillStyle = '#666';
                ctx.textAlign = 'center';
                ctx.fillText('No pLDDT scores available for this structure', chartCanvas.width / 2, chartCanvas.height / 2);
            }
        }

        // 4. Render Topology Table
        // Define common update handler
        const handleSegmentChange = (chainId, newSegments) => {
            console.log(`Segment change detected for chain ${chainId}`, newSegments);

            // Update structure object reference (updates in-memory state)
            if (structure.segments) {
                structure.segments[chainId] = newSegments;
            }

            // Re-render topology table (recurses here, but that is fine/necessary to refresh inputs)
            renderTopologyTable('topologyTableContainer', structure.segments, handleSegmentChange);

            // Re-render 3D viewer to reflect changes
            showStructurePreview(
                structure.file_url,
                structure.filename,
                structure.sequences,
                structure.segments
            );

            // Re-render Plot to update lines
            if (hasPlddt) {
                renderPlddtChart(
                    'plddtChartLanding',
                    structure.plddt,
                    structure.segments,
                    { interactive: true, onSegmentChange: handleSegmentChange }
                );
            }
        };

        renderTopologyTable('topologyTableContainer', structure.segments || {}, handleSegmentChange);

    } catch (error) {
        console.error('Error in visualizeStructure:', error);
    }
}
