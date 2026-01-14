import { showError, showInfo } from './ui.js';

// Global NGL viewer state
let nglLoaded = false;
let nglLoadPromise = null;
let currentViewer = null;

/**
 * Ensure NGL is loaded and available
 * @returns {Promise<boolean>} True if NGL is available, false otherwise
 */
async function ensureNGLLoaded() {
    // If already loaded and verified, return immediately
    if (nglLoaded && window.NGL) {
        return true;
    }

    // If loading is in progress, wait for it
    if (nglLoadPromise) {
        return await nglLoadPromise;
    }

    // Check if NGL is already available (loaded via script tag)
    if (window.NGL && typeof window.NGL.Stage === 'function') {
        console.log('NGL already loaded via script tag');
        nglLoaded = true;
        return true;
    }

    // If NGL script tag exists but not loaded yet, wait for it
    const nglScript = document.querySelector('script[src*="ngl"]');
    if (nglScript && !window.NGL) {
        console.log('Waiting for NGL script to load...');
        nglLoadPromise = waitForNGLScript();
        return await nglLoadPromise;
    }

    // Start loading NGL dynamically as fallback
    console.log('Loading NGL dynamically as fallback...');
    nglLoadPromise = loadNGLDynamically();
    return await nglLoadPromise;
}

/**
 * Wait for NGL script tag to load
 * @returns {Promise<boolean>} True if loaded successfully, false otherwise
 */
function waitForNGLScript() {
    return new Promise((resolve) => {
        let attempts = 0;
        const maxAttempts = 50; // 5 seconds with 100ms intervals

        const checkNGL = () => {
            attempts++;
            if (window.NGL && typeof window.NGL.Stage === 'function') {
                console.log('NGL script loaded successfully');
                nglLoaded = true;
                resolve(true);
            } else if (attempts >= maxAttempts) {
                console.warn('Timeout waiting for NGL script to load');
                resolve(false);
            } else {
                setTimeout(checkNGL, 100);
            }
        };

        checkNGL();
    });
}

/**
 * Load NGL library dynamically
 * @returns {Promise<boolean>} True if loaded successfully, false otherwise
 */
function loadNGLDynamically() {
    return new Promise((resolve) => {
        console.log('Loading NGL viewer dynamically...');

        // Check if we already have an NGL script to avoid duplicates
        const existingScript = document.querySelector('script[src*="ngl"]');
        if (existingScript) {
            console.log('NGL script already exists, not adding duplicate');
            resolve(false);
            return;
        }

        const script = document.createElement('script');
        script.src = 'https://unpkg.com/ngl@latest/dist/ngl.js';
        script.onload = () => {
            console.log('NGL viewer loaded dynamically');
            nglLoaded = true;
            resolve(true);
        };
        script.onerror = (error) => {
            console.error('Failed to load NGL viewer:', error);
            showError('3D viewer unavailable - continuing without structure preview');
            nglLoaded = false;
            resolve(false);
        };

        // Set a timeout to avoid hanging
        setTimeout(() => {
            if (!nglLoaded) {
                console.warn('NGL loading timeout - continuing without 3D viewer');
                script.remove();
                resolve(false);
            }
        }, 10000); // 10 second timeout

        document.head.appendChild(script);
    });
}

// Track currently loaded file
let currentFileUrl = null;
let currentComponent = null;

/**
 * Show structure preview in the 3D viewer
 * @param {string} fileUrl - URL to the structure file
 * @param {string} filename - Name of the file for display
 * @param {Array} sequences - Array of sequence objects with FP data (optional)
 * @returns {Promise<boolean>} True if preview was shown, false otherwise
 */
export async function showStructurePreview(fileUrl, filename = 'structure', sequences = null) {
    console.log('Attempting to show structure preview for:', filename);
    if (sequences) {
        console.log('FP data provided for coloring:', sequences.map(s => ({ chain: s.chain, fps: s.fps })));
    }

    const viewerContainer = document.getElementById('pdbViewerContainer');
    const viewerElement = document.getElementById('pdbViewer');

    if (!viewerContainer || !viewerElement) {
        console.warn('Viewer container elements not found - 3D preview not available');
        return false;
    }

    // Check if we can just update representations without reloading
    if (currentViewer && currentFileUrl === fileUrl && currentComponent) {
        console.log('File already loaded, updating representations only');
        updateRepresentations(currentComponent, sequences);
        return true;
    }

    try {
        // Ensure NGL is loaded
        const nglAvailable = await ensureNGLLoaded();
        if (!nglAvailable) {
            console.log('NGL not available, hiding viewer container');
            viewerContainer.style.display = 'none';
            return false;
        }

        // Clear any existing viewer
        if (currentViewer) {
            currentViewer.dispose();
            currentViewer = null;
            currentComponent = null;
            currentFileUrl = null;
        }

        // Clear the viewer element
        viewerElement.innerHTML = '';

        // Show the container
        viewerContainer.style.display = 'block';

        // Check WebGL availability before creating stage
        if (!isWebGLAvailable()) {
            console.warn('WebGL not available for structure preview');
            viewerElement.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; 
                            color: #6c757d; text-align: center; flex-direction: column;">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <div>3D preview requires WebGL</div>
                    <small class="mt-1">Please enable hardware acceleration in your browser</small>
                </div>
            `;
            return false;
        }

        // Create new NGL stage
        currentViewer = new window.NGL.Stage(viewerElement, {
            backgroundColor: '#f8f9fa',
            quality: 'medium',
            sampleLevel: 1
        });

        // Add loading indicator
        viewerElement.style.position = 'relative';
        const loadingDiv = document.createElement('div');
        loadingDiv.innerHTML = `
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        background: rgba(255,255,255,0.9); padding: 20px; border-radius: 8px; 
                        text-align: center; z-index: 1000;">
                <div class="spinner-border spinner-border-sm me-2"></div>
                Loading structure...
            </div>
        `;
        viewerElement.appendChild(loadingDiv);

        // Load the structure
        const component = await currentViewer.loadFile(fileUrl, {
            ext: getFileExtension(filename),
            name: filename
        });

        currentComponent = component;
        currentFileUrl = fileUrl;

        // Remove loading indicator
        loadingDiv.remove();

        // Apply coloring
        updateRepresentations(component, sequences);

        // Auto-view the structure
        currentViewer.autoView();

        // Handle window resize
        const resizeObserver = new ResizeObserver(() => {
            if (currentViewer) {
                currentViewer.handleResize();
            }
        });
        resizeObserver.observe(viewerElement);

        console.log('Structure preview loaded successfully');
        showInfo(`3D preview loaded for ${filename}`);
        return true;

    } catch (error) {
        console.error('Error loading structure preview:', error);

        // Show error in viewer
        viewerElement.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; 
                        color: #6c757d; text-align: center; flex-direction: column;">
                <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                <div>Failed to load 3D preview</div>
                <small class="mt-1">Structure file may be corrupted or unsupported</small>
            </div>
        `;

        return false;
    }
}

/**
 * Update representations for the given component
 */
function updateRepresentations(component, sequences) {
    console.log('updateRepresentations called with sequences:', sequences ? sequences.length : 0);

    // Remove all existing representations
    component.removeAllRepresentations();

    // Check if we have FP data
    const hasFpData = sequences && sequences.length > 0 && sequences.some(s => s.fps && s.fps.length > 0);

    if (hasFpData) {
        console.log('Applying FP-based coloring via updateRepresentations');

        // First add a base gray representation for non-FP regions
        component.addRepresentation('cartoon', {
            color: '#BDBDBD',
            quality: 'medium'
        });

        // Then add colored representations for each FP region
        sequences.forEach(seq => {
            if (seq.fps && seq.fps.length > 0) {
                console.log(`Processing ${seq.fps.length} FPs/regions for sequence ${seq.id}`);
                seq.fps.forEach(fp => {
                    // Build selection string for this FP region
                    // Handle missing chain ID
                    const chainSele = seq.chain ? `:${seq.chain} and` : '';
                    const selection = `${chainSele} ${fp.start + 1}-${fp.end + 1}`;

                    // Map FP color to hex
                    let hexColor;
                    if (fp.color === 'green') {
                        hexColor = '#66BB6A';
                    } else if (fp.color === 'magenta') {
                        hexColor = '#EC407A';
                    } else if (fp.color === 'yellow') {
                        hexColor = '#FFEE58';
                    } else {
                        hexColor = '#BDBDBD';
                    }

                    console.log(`Adding FP region: ${fp.name} selection "${selection}" color ${hexColor}`);

                    // Add representation for this FP region
                    component.addRepresentation('cartoon', {
                        sele: selection,
                        color: hexColor,
                        quality: 'medium'
                    });
                });
            }
        });
    } else {
        // No FP data, use default chainname coloring
        console.log('No FP data, applying default chainname coloring');
        component.addRepresentation('cartoon', {
            color: 'chainname',
            quality: 'medium'
        });
    }

    // Ensure the viewer updates
    if (currentViewer && currentViewer.viewer) {
        currentViewer.viewer.requestRender();
    }
}


/**
 * Get file extension from filename
 * @param {string} filename 
 * @returns {string}
 */
function getFileExtension(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    // Map common extensions to NGL-supported formats
    const extMap = {
        'pdb': 'pdb',
        'cif': 'mmcif',
        'mmcif': 'mmcif',
        'sdf': 'sdf',
        'mol2': 'mol2',
        'gro': 'gro',
        'xyz': 'xyz'
    };
    return extMap[ext] || 'pdb'; // Default to PDB
}

/**
 * Hide the structure viewer
 */
export function hideStructureViewer() {
    const viewerContainer = document.getElementById('pdbViewerContainer');
    if (viewerContainer) {
        viewerContainer.style.display = 'none';
    }

    if (currentViewer) {
        currentViewer.dispose();
        currentViewer = null;
        currentComponent = null;
        currentFileUrl = null;
    }
}

/**
 * Check if NGL viewer is available
 * @returns {boolean}
 */
export function isViewerAvailable() {
    return nglLoaded && window.NGL;
}

/**
 * Check if WebGL is available in the browser
 * @returns {boolean} True if WebGL is available
 */
function isWebGLAvailable() {
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        return !!(gl && gl.getExtension);
    } catch (e) {
        return false;
    }
}

/**
 * Show a user-friendly message about WebGL unavailability
 */
function showWebGLUnavailableMessage() {
    const message = `
        <div class="alert alert-warning" role="alert">
            <h6><i class="fas fa-exclamation-triangle me-2"></i>3D Viewer Unavailable</h6>
            <p class="mb-2">The 3D structure viewer requires WebGL, which is not available in your browser.</p>
            <details>
                <summary class="text-muted" style="cursor: pointer;">How to enable WebGL</summary>
                <ul class="mt-2 mb-0 text-muted small">
                    <li><strong>Chrome/Edge:</strong> Go to chrome://settings/system and enable "Use hardware acceleration when available"</li>
                    <li><strong>Firefox:</strong> Go to about:config and set webgl.disabled to false</li>
                    <li><strong>Safari:</strong> Enable "WebGL" in Develop menu (if available)</li>
                    <li>Ensure your graphics drivers are up to date</li>
                </ul>
            </details>
        </div>
    `;

    // Try to show in a dedicated area, fallback to console
    const alertContainer = document.querySelector(DOM.alertContainer);
    if (alertContainer) {
        // Insert at the beginning of the container
        alertContainer.insertAdjacentHTML('afterbegin', message);
    } else {
        console.warn('WebGL not available. 3D viewer disabled.');
        // Also try to show via the UI error function
        showError('3D viewer requires WebGL - please enable hardware acceleration in your browser');
    }
}

/**
 * Initialize viewer on page load (optional)
 */
export async function initializeViewer() {
    console.log('Initializing 3D viewer...');

    try {
        // First check if WebGL is available
        if (!isWebGLAvailable()) {
            console.warn('WebGL not available - 3D viewer requires WebGL support');
            showWebGLUnavailableMessage();
            return false;
        }

        const available = await ensureNGLLoaded();
        if (available) {
            console.log('3D viewer ready - NGL version:', window.NGL?.version || 'unknown');

            // Test that we can create a stage
            const testElement = document.createElement('div');
            testElement.style.width = '100px';
            testElement.style.height = '100px';
            testElement.style.position = 'absolute';
            testElement.style.left = '-9999px'; // Hide off-screen
            document.body.appendChild(testElement);

            try {
                const testStage = new window.NGL.Stage(testElement, {
                    backgroundColor: '#f8f9fa',
                    quality: 'low', // Use low quality for testing
                    sampleLevel: 0
                });

                // Test successful - clean up
                if (testStage && typeof testStage.dispose === 'function') {
                    testStage.dispose();
                }
                document.body.removeChild(testElement);
                console.log('3D viewer test successful');
                return true;
            } catch (testError) {
                console.error('3D viewer test failed:', testError);
                document.body.removeChild(testElement);

                // Check if it's a WebGL-related error
                if (testError.message && testError.message.includes('WebGL')) {
                    showWebGLUnavailableMessage();
                } else {
                    showError('3D viewer initialization failed - continuing without structure preview');
                }
                return false;
            }
        } else {
            console.log('3D viewer not available');
        }
        return available;
    } catch (error) {
        console.error('Error initializing 3D viewer:', error);
        return false;
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (currentViewer) {
        currentViewer.dispose();
    }
});
