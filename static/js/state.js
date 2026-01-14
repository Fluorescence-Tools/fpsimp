export const state = {
    uploadId: null,
    uploadedSequences: [],
    currentSequence: null,
    membraneRegions: {},
    isSelecting: false,
    startPos: -1,
    // PDB upload context
    isPdbUpload: false,
    pdbPath: null,
    derivedFasta: null,
    mergedHeader: null,
    fpSites: {
        donor: { site1: null, site2: null },
        acceptor: { site1: null, site2: null }
    },
    currentJobId: null,
    lastJobResultsUrl: null,
    statusInterval: null,
    // Structure management
    structures: [],
    selectedStructureIndex: -1,
    
    // Structure management methods
    addStructure(structureData) {
        const structure = {
            id: this.structures.length,
            upload_id: structureData.upload_id,
            filename: structureData.filename,
            file_path: structureData.file_path,
            file_url: structureData.file_url,
            file_size: structureData.file_size,
            sequences: structureData.sequences || [],
            type: structureData.type || 'pdb',
            selected: typeof structureData.selected === 'boolean' ? structureData.selected : true,
            timestamp: new Date().toISOString(),
            original_filename: structureData.original_filename, // Track original file for multi-seq FASTA
            sequence_index: structureData.sequence_index || 0,
            uniprot_id: structureData.uniprot_id || null // Store UniProt ID for PDB structures
        };
        
        this.structures.push(structure);
        this.selectedStructureIndex = this.structures.length - 1;
        
        // Update upload context for compatibility
        this.uploadId = structure.upload_id;
        this.isPdbUpload = true;
        this.pdbPath = structure.file_path;
        
        // Update sequences if provided
        if (structure.sequences && structure.sequences.length > 0) {
            this.uploadedSequences = structure.sequences;
            this.currentSequence = structure.sequences[0];
        }
        
        return structure;
    },
    
    removeStructure(index) {
        if (index >= 0 && index < this.structures.length) {
            this.structures.splice(index, 1);
            
            // Update indices
            this.structures.forEach((struct, i) => {
                struct.id = i;
            });
            
            // Update selected index
            if (this.selectedStructureIndex >= this.structures.length) {
                this.selectedStructureIndex = this.structures.length - 1;
            }
            
            // Clear upload context if no structures remain
            if (this.structures.length === 0) {
                this.clearStructures();
            }
        }
    },
    
    toggleStructureSelection(index) {
        if (this.structures[index]) {
            this.structures[index].selected = !this.structures[index].selected;
        }
    },
    
    getSelectedStructures() {
        return this.structures.filter(struct => struct.selected);
    },
    
    clearStructures() {
        this.structures = [];
        this.selectedStructureIndex = -1;
        this.isPdbUpload = false;
        this.pdbPath = null;
        this.uploadId = null;
        this.uploadedSequences = [];
        this.currentSequence = null;
    },
    
    getSelectedStructure() {
        if (this.selectedStructureIndex >= 0 && this.selectedStructureIndex < this.structures.length) {
            return this.structures[this.selectedStructureIndex];
        }
        return null;
    },
    
    selectStructure(index) {
        if (index >= 0 && index < this.structures.length) {
            this.selectedStructureIndex = index;
            const structure = this.structures[index];
            
            // Update context
            this.uploadId = structure.upload_id;
            this.pdbPath = structure.file_path;
            
            if (structure.sequences && structure.sequences.length > 0) {
                this.uploadedSequences = structure.sequences;
                this.currentSequence = structure.sequences[0];
            }
            
            return structure;
        }
        return null;
    },

    getSelectedStructures() {
        return (this.structures || []).filter(s => s.selected);
    }
};
