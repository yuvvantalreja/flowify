import { pipeline, env } from 'https://unpkg.com/@xenova/transformers@2.16.0/dist/transformers.min.js';


env.allowLocalModels = false;
env.useBrowserCache = true;


const videoPlayer = document.getElementById('videoPlayer');
const fileInput = document.getElementById('fileInput');
const transcribeButton = document.getElementById('transcribeButton');
const analyzeButton = document.getElementById('analyzeButton');
const transcriptionDiv = document.getElementById('transcription');
const topicResultsDiv = document.getElementById('topicResults');
const statusDiv = document.getElementById('status');
const modelSelector = document.getElementById('modelSelector');
const modelStatus = document.getElementById('modelStatus');

let whisperPipeline = null;
let transcriptText = '';
let mindmap;

// Original transcribeAudio function definition
async function transcribeAudio(audioData) {
    try {
        statusDiv.textContent = 'Transcribing audio...';
        transcriptionDiv.textContent = '';
        
        // Make sure we're using the correct progress element
        const progressBarFill = document.querySelector('.progress-bar-fill');
        if (progressBarFill) {
            progressBarFill.style.width = '0%';
        }

        // Check if server-side transcription is possible
        const selectedModel = modelSelector.value;
        if (selectedModel !== 'browser') {
            // Convert audio data to base64
            const base64Audio = arrayBufferToBase64(audioData.buffer);
            
            // Send to server as JSON
            const response = await fetch('/transcribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    audio: base64Audio,
                    model: selectedModel
                })
            });

            if (!response.ok) {
                throw new Error('Server transcription failed');
            }

            const result = await response.json();
            transcriptText = result.transcript || result.text || '';
            
            // Process the transcript to create segments for display
            if (transcriptText) {
                const segments = [];
                // Extract timestamps and text using regex
                const timestampPattern = /\[(\d+)\](.*?)(?=\[\d+\]|$)/g;
                let match;
                
                while ((match = timestampPattern.exec(transcriptText)) !== null) {
                    segments.push({
                        start: parseInt(match[1]),
                        text: match[2].trim()
                    });
                }
                
                // If no timestamps were found, fallback to displaying the whole text
                if (segments.length === 0 && transcriptText.trim()) {
                    segments.push({
                        start: 0,
                        text: transcriptText.trim()
                    });
                }
                
                // Display with timestamps
                console.log("Displaying transcription with segments:", segments);
                displayTranscription(segments);
            }
            
            statusDiv.textContent = 'Transcription complete!';
            analyzeButton.disabled = false;
            if (progressBarFill) {
                progressBarFill.style.width = '100%';
            }
            return;
        }
        
        // Browser-side transcription using Whisper
        if (!whisperPipeline) {
            statusDiv.textContent = 'Loading Whisper model...';
            
            // Initialize the pipeline
            whisperPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
            statusDiv.textContent = 'Model loaded. Transcribing...';
        }

        // Processing with progress updates
        const transcriptionResult = await whisperPipeline(audioData, {
            chunk_length_s: 30,
            stride_length_s: 5,
            language: 'english',
            task: 'transcribe',
            return_timestamps: true,
            callback_function: function(progressData) {
                const percentage = Math.round(progressData.progress * 100);
                statusDiv.textContent = `Transcribing: ${percentage}%`;
                
                const progressBarFill = document.querySelector('.progress-bar-fill');
                if (progressBarFill) {
                    progressBarFill.style.width = `${percentage}%`;
                }
            }
        });

        // Save and display transcription
        transcriptText = transcriptionResult.text;
        displayTranscription(transcriptionResult.chunks || []);
        
        statusDiv.textContent = 'Transcription complete!';
        analyzeButton.disabled = false;
        if (progressBarFill) {
            progressBarFill.style.width = '100%';
        }

    } catch (error) {
        console.error('Transcription error:', error);
        statusDiv.textContent = `Error: ${error.message}`;
        throw error;
    }
}

// Helper function to display transcription with timestamps
function displayTranscription(segments) {
    console.log("displayTranscription called with segments:", segments);
    
    // Clear the transcription div of all content, including the empty state
    if (transcriptionDiv) {
        transcriptionDiv.innerHTML = '';
    } else {
        console.error("transcriptionDiv element not found");
        return;
    }
    
    if (!segments || segments.length === 0) {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.innerHTML = `
            <i class="fas fa-file-alt"></i>
            <h3 class="empty-state-title">No transcription yet</h3>
            <p class="empty-state-description">Upload a video and click "Transcribe Video" to see the transcription here</p>
        `;
        transcriptionDiv.appendChild(emptyState);
        return;
    }
    
    // Create a container for the transcription content
    const transcriptionContent = document.createElement('div');
    transcriptionContent.className = 'transcription-content';
    
    segments.forEach(segment => {
        const paragraphElement = document.createElement('p');
        
        // Format and add timestamp prefix
        let timestamp = '?';
        let text = '';
        
        if (Array.isArray(segment) && segment.length >= 2) {
            // For array format [start, end, text]
            timestamp = Math.floor(segment[0]);
            text = segment[2] || '';
        } else if (typeof segment === 'object' && segment.start !== undefined) {
            // For object format {start, end, text}
            timestamp = Math.floor(segment.start);
            text = segment.text || '';
        } else {
            // Fallback
            text = segment.toString();
        }
        
        paragraphElement.textContent = `[${timestamp}] ${text}`;
        
        // Add click handler to jump to timestamp
        paragraphElement.addEventListener('click', () => {
            videoPlayer.currentTime = timestamp;
            videoPlayer.play().catch(e => console.error('Could not play video:', e));
        });
        
        paragraphElement.style.cursor = 'pointer';
        transcriptionContent.appendChild(paragraphElement);
    });
    
    transcriptionDiv.appendChild(transcriptionContent);
}

// Store original function references
const originalTranscribeAudio = transcribeAudio;
const originalAnalyzeTopics = analyzeTopics;

// Override with enhanced versions
transcribeAudio = async function(audioData) {
    // Set progress step - use the global function safely
    try {
        if (typeof window.setActiveStep === 'function') {
            window.setActiveStep('transcribe');
        }
    } catch (err) {
        console.warn('Could not update progress steps:', err);
    }
    
    try {
        await originalTranscribeAudio(audioData);
        // If successful, make analyze button fancy with pulse animation
        analyzeButton.classList.add('animate-pulse');
        setTimeout(() => analyzeButton.classList.remove('animate-pulse'), 3000);
    } catch (error) {
        console.error('Transcription error:', error);
        throw error;
    }
};

analyzeTopics = async function() {
    // Set progress step - use the global function safely
    try {
        if (typeof window.setActiveStep === 'function') {
            window.setActiveStep('analyze');
        }
    } catch (err) {
        console.warn('Could not update progress steps:', err);
    }
    
    // Remove pulse animation if still there
    analyzeButton.classList.remove('animate-pulse');
    
    try {
        await originalAnalyzeTopics();
        // If successful, we move to visualization
        try {
            if (typeof window.setActiveStep === 'function') {
                window.setActiveStep('visualize');
            }
        } catch (err) {
            console.warn('Could not update progress steps:', err);
        }
    } catch (error) {
        console.error('Analysis error:', error);
        throw error;
    }
};

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const videoURL = URL.createObjectURL(file);
        videoPlayer.src = videoURL;
        transcribeButton.disabled = false;
        analyzeButton.disabled = true;
        transcriptionDiv.textContent = '';
        topicResultsDiv.textContent = '';
        transcriptText = '';
        statusDiv.textContent = 'Video loaded. Ready to transcribe.';
        
        const progressBarFill = document.querySelector('.progress-bar-fill');
        if (progressBarFill) {
            progressBarFill.style.width = '0%';
        }
    }
});

// Audio Processing Functions
function convertAudioBuffer(audioBuffer) {
    const audioData = audioBuffer.getChannelData(0);
    const targetSampleRate = 16000;
    const resamplingRatio = targetSampleRate / audioBuffer.sampleRate;
    const resampledLength = Math.floor(audioData.length * resamplingRatio);
    const resampledData = new Float32Array(resampledLength);

    for (let i = 0; i < resampledLength; i++) {
        const originalIndex = Math.floor(i / resamplingRatio);
        resampledData[i] = audioData[originalIndex];
    }

    return resampledData;
}

async function extractAudio(videoFile) {
    try {
        console.log('Starting audio extraction');
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await videoFile.arrayBuffer();
        console.log('Video file loaded into buffer');

        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        console.log('Audio decoded successfully');

        return convertAudioBuffer(audioBuffer);
    } catch (error) {
        console.error('Error extracting audio:', error);
        throw new Error(`Failed to extract audio: ${error.message}`);
    }
}

// Define setActiveStep globally before any other functions use it
window.setActiveStep = function(step) {
    // Will be initialized properly later, but create empty function to prevent errors
    console.log(`Progress step: ${step}`);
};

// Helper function to convert ArrayBuffer to base64
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

function formatTimestamp(seconds) {
    // Round down to nearest second and return as integer
    return `[${Math.floor(seconds)}]`;
}

async function analyzeTopics() {
    try {
        statusDiv.textContent = 'Analyzing topics...';
        
        // Make sure we're using the correct progress element
        const progressBarFill = document.querySelector('.progress-bar-fill');
        if (progressBarFill) {
            progressBarFill.style.width = '0%';
        }

        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ transcript: transcriptText })
        });

        if (!response.ok) {
            throw new Error('Failed to analyze topics');
        }

        const data = await response.json();
        displayTopicResults(data.segments);
        statusDiv.textContent = 'Topic analysis complete!';
        
        if (progressBarFill) {
            progressBarFill.style.width = '100%';
        }

    } catch (error) {
        console.error('Topic analysis error:', error);
        statusDiv.textContent = `Error: ${error.message}`;
    }
}

function displayTopicResults(segments) {
    // Clear previous results
    topicResultsDiv.innerHTML = '';
    
    // Display topics in the sidebar
    segments.forEach(segment => {
        const topicElement = document.createElement('div');
        topicElement.className = 'topic';
        
        const headerElement = document.createElement('div');
        headerElement.className = 'topic-header';
        headerElement.textContent = segment.topic_name;
        
        const contentElement = document.createElement('div');
        contentElement.className = 'topic-content';
        
        // Handle content that could be either an array or a string
        if (Array.isArray(segment.content)) {
            contentElement.textContent = segment.content.join('\n');
        } else {
            contentElement.textContent = segment.content;
        }
        
        topicElement.appendChild(headerElement);
        topicElement.appendChild(contentElement);
        
        // If this is a parent topic with children, display them as nested elements
        if (segment.is_parent && segment.children && segment.children.length > 0) {
            topicElement.classList.add('parent-topic');
            
            const childrenContainer = document.createElement('div');
            childrenContainer.className = 'subtopics';
            
            segment.children.forEach(child => {
                const childElement = document.createElement('div');
                childElement.className = 'subtopic';
                
                const childHeader = document.createElement('div');
                childHeader.className = 'subtopic-header';
                childHeader.textContent = child.topic_name;
                
                const childContent = document.createElement('div');
                childContent.className = 'subtopic-content';
                
                // Handle content that could be either an array or a string
                if (Array.isArray(child.content)) {
                    childContent.textContent = child.content.join('\n');
                } else {
                    childContent.textContent = child.content;
                }
                
                childElement.appendChild(childHeader);
                childElement.appendChild(childContent);
                childrenContainer.appendChild(childElement);
            });
            
            topicElement.appendChild(childrenContainer);
        }
        
        topicResultsDiv.appendChild(topicElement);
    });

    // Update mindmap and show fullscreen overlay
    mindmap.update(segments);
}

// Event Listeners
transcribeButton.addEventListener('click', async () => {
    try {
        transcribeButton.disabled = true;
        modelSelector.disabled = true;
        const videoFile = fileInput.files[0];

        if (!videoFile) {
            throw new Error('No video file selected');
        }
        
        const videoPreview = document.querySelector('.video-preview');
        const videoHeight = videoPlayer.offsetHeight;
        if (videoHeight > 0) {
            videoPreview.style.height = `${videoHeight}px`;
        }

        statusDiv.textContent = 'Extracting audio...';
        const audioData = await extractAudio(videoFile);
        await transcribeAudio(audioData);
        modelSelector.disabled = false;

    } catch (error) {
        console.error('Processing error:', error);
        statusDiv.textContent = `Error: ${error.message}`;
        transcribeButton.disabled = false;
        modelSelector.disabled = false;
    }
});

analyzeButton.addEventListener('click', analyzeTopics);

modelSelector.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    transcribeButton.disabled = false;
    analyzeButton.disabled = true;
    await initWhisper(selectedModel);
    if (fileInput.files.length > 0) {
        transcribeButton.disabled = false;
    }
});

// Initialize with the selected model when the page loads
window.addEventListener('DOMContentLoaded', () => {
    // Initialize all UI components
    initThemeToggle();
    initFAB();
    initGallery();
    initCustomVideoPlayer();
    initUploadParticles();
    initProgressSteps();
    
    // Original initialization
    mindmap = new TopicMindmap();
    initWhisper(modelSelector.value).catch(error => {
        console.error('Failed to initialize Whisper:', error);
        statusDiv.textContent = 'Failed to initialize the transcription model';
    });
});

// Theme toggle functionality
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const htmlElement = document.documentElement;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        htmlElement.setAttribute('data-theme', savedTheme);
    }
    
    themeToggle.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        htmlElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
}

// FAB functionality
function initFAB() {
    const fabContainer = document.getElementById('fabContainer');
    const mainFab = document.getElementById('mainFab');
    const exportPdfBtn = document.getElementById('exportPdfBtn');
    const exportPngBtn = document.getElementById('exportPngBtn');
    const exportSvgBtn = document.getElementById('exportSvgBtn');
    const shareMindmapBtn = document.getElementById('shareMindmapBtn');
    
    // Toggle FAB options
    mainFab.addEventListener('click', () => {
        fabContainer.classList.toggle('active');
    });
    
    // Close FAB when clicking outside
    document.addEventListener('click', (e) => {
        if (!fabContainer.contains(e.target) && fabContainer.classList.contains('active')) {
            fabContainer.classList.remove('active');
        }
    });
    
    // Export as PDF
    exportPdfBtn.addEventListener('click', () => {
        exportMindmap('pdf');
    });
    
    // Export as PNG
    exportPngBtn.addEventListener('click', () => {
        exportMindmap('png');
    });
    
    // Export as SVG
    exportSvgBtn.addEventListener('click', () => {
        exportMindmap('svg');
    });
    
    // Share mindmap
    shareMindmapBtn.addEventListener('click', () => {
        shareMindmap();
    });
}

// Gallery functionality
function initGallery() {
    const galleryContainer = document.getElementById('resultsGallery');
    const galleryClose = document.getElementById('galleryClose');
    const galleryContent = document.getElementById('galleryContent');
    
    // Add "Recent Mindmaps" option to FAB
    const fabContainer = document.getElementById('fabContainer');
    const recentMindmapsBtn = document.createElement('button');
    recentMindmapsBtn.className = 'fab-option';
    recentMindmapsBtn.setAttribute('title', 'Recent Mindmaps');
    recentMindmapsBtn.innerHTML = '<i class="fas fa-history"></i>';
    fabContainer.querySelector('.fab-options').appendChild(recentMindmapsBtn);
    
    // Show gallery
    recentMindmapsBtn.addEventListener('click', () => {
        showGallery();
        fabContainer.classList.remove('active');
    });
    
    // Close gallery
    galleryClose.addEventListener('click', () => {
        galleryContainer.classList.remove('active');
    });
    
    // Close gallery with escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && galleryContainer.classList.contains('active')) {
            galleryContainer.classList.remove('active');
        }
    });
}

// Show gallery and populate with saved mindmaps
function showGallery() {
    const galleryContainer = document.getElementById('resultsGallery');
    const galleryContent = document.getElementById('galleryContent');
    
    // Clear existing content
    galleryContent.innerHTML = '';
    
    // Get saved mindmaps from localStorage
    const savedMindmaps = getSavedMindmaps();
    
    if (savedMindmaps.length === 0) {
        // Show empty state
        galleryContent.innerHTML = `
            <div class="gallery-empty">
                <i class="fas fa-file-alt" style="font-size: 3rem; color: var(--text-secondary); margin-bottom: 1rem;"></i>
                <p>No saved mindmaps yet</p>
                <p>Your analyzed videos will appear here</p>
            </div>
        `;
    } else {
        // Create gallery items for each saved mindmap
        savedMindmaps.forEach(mindmap => {
            createGalleryItem(mindmap);
        });
    }
    
    // Show gallery
    galleryContainer.classList.add('active');
}

// Create a gallery item for a saved mindmap
function createGalleryItem(mindmapData) {
    const galleryContent = document.getElementById('galleryContent');
    
    const item = document.createElement('div');
    item.className = 'gallery-item';
    
    // Use a random color from our palette for preview
    const colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#e67e22', '#f1c40f', '#1abc9c'];
    const randomColor = colors[Math.floor(Math.random() * colors.length)];
    
    item.innerHTML = `
        <div class="gallery-item-preview" style="background-color: ${randomColor};">
            <div style="height: 100%; display: flex; align-items: center; justify-content: center; color: white;">
                <i class="fas fa-project-diagram" style="font-size: 2rem;"></i>
            </div>
        </div>
        <div class="gallery-item-info">
            <div class="gallery-item-title">${mindmapData.title || 'Untitled Mindmap'}</div>
            <div class="gallery-item-date">${new Date(mindmapData.date).toLocaleDateString()}</div>
        </div>
    `;
    
    // Load mindmap when clicked
    item.addEventListener('click', () => {
        loadMindmap(mindmapData);
        document.getElementById('resultsGallery').classList.remove('active');
    });
    
    galleryContent.appendChild(item);
}

// Export mindmap in various formats
function exportMindmap(format) {
    if (!mindmap || !mindmap.mindmap) {
        alert('No mindmap to export. Please analyze a video first.');
        return;
    }
    
    try {
        switch (format) {
            case 'pdf':
                exportAsPDF();
                break;
            case 'png':
                exportAsPNG();
                break;
            case 'svg':
                exportAsSVG();
                break;
        }
    } catch (error) {
        console.error(`Error exporting as ${format}:`, error);
        alert(`Failed to export mindmap as ${format.toUpperCase()}`);
    }
}

// Export as PDF using html2pdf
function exportAsPDF() {
    // We'll use html2pdf.js for PDF export
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js';
    document.head.appendChild(script);
    
    script.onload = () => {
        const element = document.getElementById('mindmap');
        const opt = {
            margin: 1,
            filename: 'flowify-mindmap.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'landscape' }
        };
        
        // Use html2pdf library to generate PDF
        html2pdf().set(opt).from(element).save();
    };
}

// Export as PNG
function exportAsPNG() {
    // We'll use html2canvas for PNG export
    const script = document.createElement('script');
    script.src = 'https://html2canvas.hertzen.com/dist/html2canvas.min.js';
    document.head.appendChild(script);
    
    script.onload = () => {
        html2canvas(document.getElementById('mindmap'), {
            scale: 2,
            backgroundColor: null
        }).then(canvas => {
            const link = document.createElement('a');
            link.download = 'flowify-mindmap.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        });
    };
}

// Export as SVG
function exportAsSVG() {
    alert('SVG export functionality is under development. This feature will be available in a future update.');
}

// Share mindmap (placeholder)
function shareMindmap() {
    // Save current mindmap to localStorage
    if (mindmap && mindmap.mindmap) {
        const data = mindmap.mindmap.get_data();
        const mindmapData = {
            id: Date.now(),
            title: data.data.topic || 'Video Topics',
            data: data,
            date: new Date().toISOString()
        };
        
        saveMindmap(mindmapData);
        
        alert('Mindmap saved! You can access it from the Recent Mindmaps gallery.');
    } else {
        alert('No mindmap to share. Please analyze a video first.');
    }
}

// Save mindmap to localStorage
function saveMindmap(mindmapData) {
    let savedMindmaps = getSavedMindmaps();
    savedMindmaps.push(mindmapData);
    
    // Keep only the most recent 20 mindmaps
    if (savedMindmaps.length > 20) {
        savedMindmaps = savedMindmaps.slice(-20);
    }
    
    localStorage.setItem('flowify-mindmaps', JSON.stringify(savedMindmaps));
}

// Get saved mindmaps from localStorage
function getSavedMindmaps() {
    const savedData = localStorage.getItem('flowify-mindmaps');
    return savedData ? JSON.parse(savedData) : [];
}

// Load a mindmap from saved data
function loadMindmap(mindmapData) {
    if (mindmap && mindmap.mindmap && mindmapData.data) {
        mindmap.mindmap.show(mindmapData.data);
        mindmap.mindmapOverlay.classList.add('active');
    }
}

class TopicMindmap {
    constructor() {
        this.mindmap = null;
        this.container = 'mindmap';
        this.zoomScale = 1;
        this.isPanning = false;
        this.panX = 0;
        this.panY = 0;
        this.isEditing = false;
        this.segmentData = new Map();
        
        // Get DOM elements
        this.mainVideo = document.getElementById('videoPlayer');
        this.overlayVideo = document.getElementById('overlayVideo');
        this.videoSection = document.getElementById('videoSection');
        this.videoInfo = document.getElementById('videoInfo');
        this.toolbar = document.querySelector('.mindmap-toolbar');
        this.mindmapOverlay = document.getElementById('mindmapOverlay');
        this.videoCloseButton = document.getElementById('videoCloseButton');
        
        // Context menu elements
        this.contextMenu = document.getElementById('nodeContextMenu');
        this.colorPicker = document.getElementById('colorPicker');
        this.currentNodeId = null;
        
        // Initially hide video section
        if (this.videoSection) {
            this.videoSection.style.display = 'none';
        }
        
        this.initialize();
        this.setupToolbar();
        this.setupNodeInteraction();
        this.setupVideoCloseButton();
        this.setupContextMenu();
    }

    initialize() {
        const options = {
            container: this.container,
            theme: 'primary',
            editable: true,
            support_html: true,
            view: {
                draggable: true,
                hmargin: 300,
                vmargin: 200,
                line_width: 2,
                line_color: '#95a5a6',
                node_spacing: 100
            },
            layout: {
                hspace: 150,
                vspace: 80,
                pspace: 30
            }
        };

        try {
            // Add a style element for immediate expander fixes
            this.addExpanderFixStyles();
            
            this.mindmap = new jsMind(options);
            const emptyMind = this.createEmptyMindMap();
            this.mindmap.show(emptyMind);
            
            const container = document.querySelector('.fullscreen-mindmap');
            container.style.transformOrigin = '0 0';
            container.style.position = 'absolute';
            container.style.width = '100%';
            container.style.height = '100%';
            
            const mindmapWrapper = document.getElementById('mindmap');
            mindmapWrapper.style.position = 'absolute';
            mindmapWrapper.style.width = '100%';
            mindmapWrapper.style.height = '100%';
            mindmapWrapper.style.overflow = 'visible';
            
            // Apply custom classes to expanders after rendering
            this.applyCustomExpanderClasses();
            
            this.setupNodeInteraction();
            this.setupPanning();
        } catch (error) {
            console.error('Error initializing mindmap:', error);
        }
    }

    addExpanderFixStyles() {
        // Add a style element to ensure expanders don't show any text content
        const style = document.createElement('style');
        style.textContent = `
            jmexpander * {
                display: none !important;
            }
            jmexpander::before, jmexpander::after {
                display: none !important;
            }
        `;
        document.head.appendChild(style);
    }

    setupNodeInteraction() {
        // Remove any existing event listeners
        const container = document.getElementById(this.container);
        if (!container) return;

        container.addEventListener('click', (e) => {
            const target = e.target;
            if (target.tagName.toLowerCase() === 'jmnode') {
                const nodeId = target.getAttribute('nodeid');
                if (nodeId) {
                    this.handleNodeClick(nodeId);
                }
            }
        });
    }

    handleNodeClick(nodeId) {
        console.log('Node clicked:', nodeId);
        const segmentInfo = this.segmentData.get(nodeId);
        console.log('Segment info:', segmentInfo);
        
        if (segmentInfo && segmentInfo.timestamp !== undefined) {
            // Show video section
            if (this.videoSection) {
                this.videoSection.style.display = 'block';
                console.log('Video section displayed');
            }
            
            // Adjust layout
            if (this.toolbar) {
                this.toolbar.classList.add('with-video');
            }
            
            // Setup video
            if (this.mainVideo && this.mainVideo.src && this.overlayVideo) {
                console.log('Setting up video playback at time:', segmentInfo.timestamp);
                
                // Set video source if needed
                if (this.overlayVideo.src !== this.mainVideo.src) {
                    this.overlayVideo.src = this.mainVideo.src;
                }
                
                // Get the timestamp as a number
                let timeInSeconds = segmentInfo.timestamp;
                
                // If it's still a string (with brackets), parse it
                if (typeof timeInSeconds === 'string') {
                    if (timeInSeconds.startsWith('[') && timeInSeconds.endsWith(']')) {
                        timeInSeconds = parseFloat(timeInSeconds.slice(1, -1));
                    } else {
                        timeInSeconds = parseFloat(timeInSeconds);
                    }
                }
                
                // Ensure we have a valid number
                if (!isNaN(timeInSeconds)) {
                    console.log('Seeking to time:', timeInSeconds);
                    this.overlayVideo.currentTime = timeInSeconds;
                    this.overlayVideo.play().catch(error => {
                        console.error('Error playing video:', error);
                    });
                    
                    // Update info
                    if (this.videoInfo) {
                        const timestamp = this.formatTimestamp(timeInSeconds);
                        const content = segmentInfo.content.replace(/\[\d+(?:\.\d+)?\]\s*/, '');
                        this.videoInfo.textContent = `${timestamp}: ${content}`;
                    }
                } else {
                    console.error('Invalid timestamp:', segmentInfo.timestamp);
                }
            }
            
            // Update layout
            this.updateLayout();
        }
    }

    updateLayout() {
        // Adjust mindmap section width when video is shown
        const mindmapSection = document.querySelector('.mindmap-section');
        const videoWidth = '40%';
        const mindmapWidth = '60%';
        
        if (this.videoSection && this.videoSection.style.display === 'block') {
            this.videoSection.style.width = videoWidth;
            if (mindmapSection) {
                mindmapSection.style.width = mindmapWidth;
            }
        }
        
        // Force mindmap refresh
        if (this.mindmap) {
            this.mindmap.resize();
        }
    }

    setupNodeEditing() {
        // Add double-click event listener for editing nodes
        const container = document.getElementById(this.container);
        container.addEventListener('dblclick', (e) => {
            const target = e.target;
            if (target.tagName.toLowerCase() === 'jmnode') {
                this.editNode(target);
            }
        });

        // Add node drag and drop functionality
        container.addEventListener('mousedown', (e) => {
            if (!this.isPanning && e.target.tagName.toLowerCase() === 'jmnode') {
                this.startNodeDrag(e);
            }
        });
    }

    editNode(nodeElement) {
        if (this.isPanning) return;
        
        const nodeId = nodeElement.getAttribute('nodeid');
        const currentText = this.mindmap.get_node(nodeId).topic;
        
        // Create and style the input element
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentText;
        input.style.width = Math.max(nodeElement.offsetWidth, 100) + 'px';
        input.style.height = nodeElement.offsetHeight + 'px';
        input.style.position = 'absolute';
        input.style.left = nodeElement.offsetLeft + 'px';
        input.style.top = nodeElement.offsetTop + 'px';
        input.style.zIndex = '1000';
        input.style.padding = '4px';
        input.style.border = '2px solid #3498db';
        input.style.borderRadius = '4px';
        input.style.backgroundColor = 'white';
        
        // Replace node with input
        nodeElement.style.visibility = 'hidden';
        nodeElement.parentNode.appendChild(input);
        input.focus();
        input.select();

        const finishEditing = () => {
            const newText = input.value.trim();
            if (newText) {
                this.mindmap.update_node(nodeId, newText);
            }
            nodeElement.style.visibility = 'visible';
            input.remove();
        };

        input.addEventListener('blur', finishEditing);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                finishEditing();
            } else if (e.key === 'Escape') {
                nodeElement.style.visibility = 'visible';
                input.remove();
            }
        });
    }

    startNodeDrag(e) {
        if (this.isPanning) return;

        const nodeElement = e.target;
        const nodeId = nodeElement.getAttribute('nodeid');
        if (!nodeId) return;

        const startX = e.clientX;
        const startY = e.clientY;
        const originalLeft = nodeElement.offsetLeft;
        const originalTop = nodeElement.offsetTop;

        const mouseMoveHandler = (e) => {
            const dx = (e.clientX - startX) / this.zoomScale;
            const dy = (e.clientY - startY) / this.zoomScale;
            
            nodeElement.style.left = (originalLeft + dx) / this.zoomScale + 'px';
            nodeElement.style.top = (originalTop + dy) / this.zoomScale + 'px';
            
            // Update connected lines
            this.mindmap.layout.layout();
        };

        const mouseUpHandler = () => {
            document.removeEventListener('mousemove', mouseMoveHandler);
            document.removeEventListener('mouseup', mouseUpHandler);
            
            // Ensure the mindmap updates its internal positions
            this.mindmap.layout.layout();
        };

        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
    }

    setupPanning() {
        const container = document.querySelector('.fullscreen-mindmap');
        const mindmapWrapper = document.getElementById('mindmap');
        const panButton = document.getElementById('panButton');
        
        panButton.addEventListener('click', () => {
            this.isPanning = !this.isPanning;
            panButton.classList.toggle('active');
            container.style.cursor = this.isPanning ? 'grab' : 'default';
        });

        container.addEventListener('wheel', (e) => {
            if (e.ctrlKey) {
                e.preventDefault();
                const delta = e.deltaY * -0.01;
                const newScale = Math.min(Math.max(this.zoomScale + delta, 0.1), 3);
                
                // Get mouse position relative to container
                const rect = container.getBoundingClientRect();
                const x = (e.clientX - rect.left) / this.zoomScale;
                const y = (e.clientY - rect.top) / this.zoomScale;
                
                // Calculate new pan position to zoom towards mouse
                const scale = newScale / this.zoomScale;
                this.panX = x - (x - this.panX) * scale;
                this.panY = y - (y - this.panY) * scale;
                
                this.zoomScale = newScale;
                this.applyTransform();
            }
        });

        container.addEventListener('mousedown', (e) => {
            if (!this.isPanning) return;
            
            container.style.cursor = 'grabbing';
            const startX = e.clientX - this.panX * this.zoomScale;
            const startY = e.clientY - this.panY * this.zoomScale;
            
            const mouseMoveHandler = (e) => {
                if (!this.isPanning) return;
                
                this.panX = (e.clientX - startX) / this.zoomScale;
                this.panY = (e.clientY - startY) / this.zoomScale;
                this.applyTransform();
            };
            
            const mouseUpHandler = () => {
                container.style.cursor = this.isPanning ? 'grab' : 'default';
                document.removeEventListener('mousemove', mouseMoveHandler);
                document.removeEventListener('mouseup', mouseUpHandler);
            };
            
            document.addEventListener('mousemove', mouseMoveHandler);
            document.addEventListener('mouseup', mouseUpHandler);
        });
    }

    applyTransform() {
        const container = document.querySelector('.fullscreen-mindmap');
        const matrix = `matrix(${this.zoomScale}, 0, 0, ${this.zoomScale}, ${this.panX * this.zoomScale}, ${this.panY * this.zoomScale})`;
        container.style.transform = matrix;
    }

    resetView() {
        this.zoomScale = 1;
        this.panX = 0;
        this.panY = 0;
        this.applyTransform();
        
        const panButton = document.getElementById('panButton');
        this.isPanning = false;
        panButton.classList.remove('active');
        document.querySelector('.fullscreen-mindmap').style.cursor = 'default';
        
        if (this.mindmap) {
            this.mindmap.resize();
            
            // Center the mindmap after reset
            const container = document.querySelector('.fullscreen-mindmap');
            const mindmapEl = document.getElementById(this.container);
            if (container && mindmapEl) {
                const containerRect = container.getBoundingClientRect();
                const mindmapRect = mindmapEl.getBoundingClientRect();
                
                this.panX = (containerRect.width - mindmapRect.width) / 2;
                this.panY = (containerRect.height - mindmapRect.height) / 2;
                this.applyTransform();
            }
        }
    }

    setupToolbar() {
        const backButton = document.getElementById('backButton');
        const editButton = document.getElementById('editButton');
        const zoomInButton = document.getElementById('zoomInButton');
        const zoomOutButton = document.getElementById('zoomOutButton');
        const resetButton = document.getElementById('resetButton');
        const panButton = document.getElementById('panButton');

        if (backButton) {
            backButton.addEventListener('click', () => {
                // Hide video section when closing
                this.videoSection.classList.remove('active');
                this.toolbar.classList.remove('with-video');
                
                // Stop video playback
                if (this.overlayVideo.src) {
                    this.overlayVideo.pause();
                }
                
                document.getElementById('mindmapOverlay').classList.remove('active');
            });

        }

        if (editButton) {
            editButton.addEventListener('click', () => {
                this.isEditing = !this.isEditing;
                editButton.classList.toggle('active');
                if (this.isEditing) {
                    this.mindmap.enable_edit();
                    editButton.innerHTML = '<i class="fas fa-edit"></i> Done';
                } else {
                    this.mindmap.disable_edit();
                    editButton.innerHTML = '<i class="fas fa-edit"></i> Edit';
                }
            });
        }

        if (zoomInButton) {
            zoomInButton.addEventListener('click', () => this.zoomIn());
        }

        if (zoomOutButton) {
            zoomOutButton.addEventListener('click', () => this.zoomOut());
        }

        if (resetButton) {
            resetButton.addEventListener('click', () => this.resetView());
        }
    }

    createEmptyMindMap() {
        return {
            meta: {
                name: 'Topics',
                author: 'Video Analyzer',
                version: '1.0'
            },
            format: 'node_tree',
            data: {
                id: 'root',
                topic: 'Video Topics',
                direction: 'center',
                children: []
            }
        };
    }

    zoomIn() {
        this.zoomScale = Math.min(this.zoomScale + 0.1, 2);
        this.applyZoom();
    }

    zoomOut() {
        this.zoomScale = Math.max(this.zoomScale - 0.1, 0.5);
        this.applyZoom();
    }

    resetView() {
        this.zoomScale = 1;
        this.applyZoom();
    }

    applyZoom() {
        const container = document.querySelector('.fullscreen-mindmap');
        if (container) {
            container.style.transform = `scale(${this.zoomScale})`;
            container.style.transformOrigin = 'center center';
        }
    }

    createMindMapData(segments) {
        if (!segments || segments.length === 0) {
            return this.createEmptyMindMap();
        }

        const mind = {
            meta: {
                name: 'Topics',
                author: 'Video Analyzer',
                version: '1.0'
            },
            format: 'node_tree',
            data: {
                id: 'root',
                topic: 'Video Topics',
                direction: 'center',
                expanded: true,
                children: []
            }
        };

        // Clear previous segment data
        this.segmentData.clear();

        // Process each segment
        let nodeId = 1;
        
        segments.forEach(segment => {
            // Extract a cleaner topic name without repeating terms
            const topicName = segment.topic_name.split(':')[1]?.trim() || segment.topic_name.trim();
            
            // Create the main topic node
            const topicNode = {
                id: `topic_${nodeId}`,
                topic: topicName,
                direction: this.getDirection(nodeId, segments.length),
                expanded: false,
                children: []
            };
            
            // Store the timestamp for the main node
            this.segmentData.set(`topic_${nodeId}`, {
                timestamp: segment.timestamp || 0,
                content: Array.isArray(segment.content) ? segment.content.join('\n') : segment.content
            });
            
            // If this is a parent topic with children, add them as subnodes
            if (segment.is_parent && segment.children && segment.children.length > 0) {
                segment.children.forEach((child, childIndex) => {
                    const childName = child.topic_name.split(':')[1]?.trim() || child.topic_name.trim();
                    const childContent = Array.isArray(child.content) ? child.content.join('\n') : child.content;
                    
                    // Extract a meaningful snippet from the content
                    let contentSnippet = '';
                    const cleanContent = childContent.replace(/\[\d+(?:\.\d+)?\]\s*/g, '');
                    contentSnippet = cleanContent.split('.')[0];
                    if (contentSnippet.length > 50) {
                        contentSnippet = contentSnippet.slice(0, 50) + '...';
                    } else {
                        contentSnippet += '...';
                    }
                    
                    const childNodeId = `content_${nodeId}_${childIndex}`;
                    
                    topicNode.children.push({
                        id: childNodeId,
                        topic: childName,
                        direction: topicNode.direction,
                        expanded: false
                    });
                    
                    // Store the timestamp and content for this child node
                    this.segmentData.set(childNodeId, {
                        timestamp: child.timestamp || 0,
                        content: childContent
                    });
                });
            } 
            // For regular segments or parents without children, add their content as subnodes
            else {
                // Get the content as an array
                const contentArray = Array.isArray(segment.content) ? segment.content : [segment.content];
                
                // If there's a lot of content, create content summary nodes
                if (contentArray.length > 5) {
                    // Create just one summary node for all content
                    const contentNodeId = `content_${nodeId}_0`;
                    
                    // Create a content summary
                    const cleanContent = contentArray.join(' ').replace(/\[\d+(?:\.\d+)?\]\s*/g, '');
                    let contentSnippet = cleanContent.split('.')[0];
                    if (contentSnippet.length > 50) {
                        contentSnippet = contentSnippet.slice(0, 50) + '...';
                    } else {
                        contentSnippet += '...';
                    }
                    
                    topicNode.children.push({
                        id: contentNodeId,
                        topic: contentSnippet,
                        direction: topicNode.direction,
                        expanded: false
                    });
                    
                    // Store the timestamp and full content
                    this.segmentData.set(contentNodeId, {
                        timestamp: segment.timestamp || 0,
                        content: contentArray.join('\n')
                    });
                } 
                // For small segments, include separate nodes for each part
                else {
                    contentArray.forEach((content, index) => {
                        if (typeof content === 'string' && content.trim()) {
                            const contentNodeId = `content_${nodeId}_${index}`;
                            
                            // Clean up the content for display
                            const cleanContent = content.replace(/\[\d+(?:\.\d+)?\]\s*/g, '');
                            let contentSnippet = cleanContent.split('.')[0];
                            if (contentSnippet.length > 50) {
                                contentSnippet = contentSnippet.slice(0, 50) + '...';
                            } else {
                                contentSnippet += '...';
                            }
                            
                            topicNode.children.push({
                                id: contentNodeId,
                                topic: contentSnippet,
                                direction: topicNode.direction,
                                expanded: false
                            });
                            
                            // Store the timestamp and full content
                            this.segmentData.set(contentNodeId, {
                                timestamp: segment.timestamp || 0,
                                content: content
                            });
                        }
                    });
                }
            }
            
            mind.data.children.push(topicNode);
            nodeId++;
        });

        return mind;
    }
    
    extractTimestamp(text) {
        // Look for timestamps in the format [X.X] or [X]
        const matches = text.match(/\[(\d+(?:\.\d+)?)\]/g);
        if (matches && matches.length > 0) {
            // Extract the first timestamp found
            const firstTimestamp = matches[0].replace('[', '').replace(']', '');
            // Round down to the nearest second (floor)
            const seconds = Math.floor(parseFloat(firstTimestamp));
            console.log('Extracted timestamp:', seconds, 'from text:', text);
            return seconds;
        }
        return 0;
    }

    getDirection(index, total) {
        const halfTotal = Math.ceil(total / 2);
        if (index <= halfTotal) {
            return 'right';
        } else {
            return 'left';
        }
    }

    update(segments) {
        if (!this.mindmap) {
            console.error('JSMind not initialized');
            return;
        }

        try {
            // Store any current node colors before updating
            const nodeColors = this.saveNodeColors();
            
            // Clear previous data
            this.segmentData.clear();
            
            // Create and show new mindmap
            const mindData = this.createMindMapData(segments);
            this.mindmap.show(mindData);
            
            // Save the mindmap to gallery
            this.saveMindmapToGallery(mindData);
            
            // Restore node colors
            this.restoreNodeColors(nodeColors);
            
            // Apply custom classes to expanders
            this.applyCustomExpanderClasses();
            
            // Reset view and show overlay
            this.resetView();
            if (this.mindmapOverlay) {
                this.mindmapOverlay.classList.add('active');
            }
            
            // Hide video section initially
            if (this.videoSection) {
                this.videoSection.style.display = 'none';
            }
            
            // Reset mindmap section to full width
            const mindmapSection = document.querySelector('.mindmap-section');
            if (mindmapSection) {
                mindmapSection.style.width = '100%';
            }
            
            // Remove video-related classes
            if (this.toolbar) {
                this.toolbar.classList.remove('with-video');
            }
            
            // Reinitialize node interaction
            this.setupNodeInteraction();
            
        } catch (error) {
            console.error('Error updating mindmap:', error);
        }
    }
    
    saveMindmapToGallery(mindData) {
        // Create mindmap data object
        const mindmapData = {
            id: Date.now(),
            title: mindData.data.topic || 'Video Topics',
            data: mindData,
            date: new Date().toISOString()
        };
        
        // Save to localStorage
        saveMindmap(mindmapData);
    }
    
    saveNodeColors() {
        // Save current node colors
        const nodeColors = {};
        if (this.mindmap) {
            const nodes = document.querySelectorAll('jmnode');
            nodes.forEach(node => {
                const nodeId = node.getAttribute('nodeid');
                if (nodeId) {
                    const colorClasses = [
                        'node-blue', 'node-green', 'node-red', 'node-purple', 
                        'node-orange', 'node-yellow', 'node-teal', 'node-gray'
                    ];
                    
                    colorClasses.forEach(cls => {
                        if (node.classList.contains(cls)) {
                            nodeColors[nodeId] = cls.replace('node-', '');
                        }
                    });
                }
            });
        }
        return nodeColors;
    }
    
    restoreNodeColors(nodeColors) {
        // Restore saved node colors
        if (nodeColors && Object.keys(nodeColors).length > 0) {
            setTimeout(() => {
                for (const [nodeId, color] of Object.entries(nodeColors)) {
                    this.changeNodeColor(nodeId, color);
                }
            }, 100);
        }
    }

    setupVideoCloseButton() {
        if (this.videoCloseButton) {
            this.videoCloseButton.addEventListener('click', () => {
                this.closeVideoOverlay();
            });
        }
    }
    
    closeVideoOverlay() {
        // Hide video section
        if (this.videoSection) {
            this.videoSection.style.display = 'none';
            
            // Stop video playback
            if (this.overlayVideo && this.overlayVideo.src) {
                this.overlayVideo.pause();
            }
        }
        
        // Remove active class from toolbar
        if (this.toolbar) {
            this.toolbar.classList.remove('with-video');
        }
        
        // Reset mindmap section to full width
        const mindmapSection = document.querySelector('.mindmap-section');
        if (mindmapSection) {
            mindmapSection.style.width = '100%';
        }
        
        // Force mindmap refresh
        if (this.mindmap) {
            this.mindmap.resize();
        }
    }

    applyCustomExpanderClasses() {
        // Add custom classes to expand/collapse buttons
        setTimeout(() => {
            this.processExpanders();
            
            // Set up a mutation observer to handle future changes
            const observer = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        this.processExpanders();
                    }
                });
            });
            
            observer.observe(document.getElementById(this.container), {
                childList: true,
                subtree: true,
                characterData: true
            });
        }, 100);
    }
    
    processExpanders() {
        const expanders = document.querySelectorAll('jmexpander');
        expanders.forEach(expander => {
            // Check if it's an expand or collapse button
            if (expander.textContent.includes('+')) {
                expander.classList.add('expand');
                // Clear content more aggressively
                expander.textContent = '';
                expander.innerText = '';
                expander.innerHTML = '';
            } else if (expander.textContent.includes('-')) {
                expander.classList.add('collapse');
                // Clear content more aggressively
                expander.textContent = '';
                expander.innerText = '';
                expander.innerHTML = '';
            }
        });
    }

    setupContextMenu() {
        // Hide context menu when clicking elsewhere
        document.addEventListener('click', (e) => {
            if (!this.contextMenu.contains(e.target)) {
                this.hideContextMenu();
            }
        });
        
        // Hide context menu on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideContextMenu();
            }
        });
        
        // Hide color picker by default and setup action handlers
        this.colorPicker.style.display = 'none';
        this.setupContextMenuActions();
        
        // Add event listeners for right-click context menu
        const container = document.getElementById(this.container);
        if (!container) return;

        container.addEventListener('contextmenu', (e) => {
            const target = e.target;
            if (target.tagName.toLowerCase() === 'jmnode') {
                e.preventDefault();
                const nodeId = target.getAttribute('nodeid');
                if (nodeId) {
                    this.currentNodeId = nodeId;
                    this.showContextMenu(e.clientX, e.clientY);
                    
                    // Update expand/collapse text based on node state
                    const expandAction = document.getElementById('expandCollapseAction');
                    const node = this.mindmap.get_node(nodeId);
                    const isExpanded = this.mindmap.get_node_expanded(node);
                    
                    if (expandAction) {
                        if (isExpanded) {
                            expandAction.innerHTML = '<i class="fas fa-angle-double-up"></i> Collapse all children';
                        } else {
                            expandAction.innerHTML = '<i class="fas fa-angle-double-down"></i> Expand all children';
                        }
                    }
                    
                    // Disable parent addition for root node
                    const addParentAction = document.getElementById('addParentAction');
                    if (addParentAction) {
                        if (nodeId === 'root') {
                            addParentAction.classList.add('disabled');
                            addParentAction.style.opacity = '0.5';
                            addParentAction.style.cursor = 'not-allowed';
                        } else {
                            addParentAction.classList.remove('disabled');
                            addParentAction.style.opacity = '1';
                            addParentAction.style.cursor = 'pointer';
                        }
                    }
                }
            }
        });
    }
    
    setupContextMenuActions() {
        // Edit node action
        document.getElementById('editNodeAction').addEventListener('click', () => {
            this.editNodeText(this.currentNodeId);
            this.hideContextMenu();
        });
        
        // Add child node action
        document.getElementById('addChildAction').addEventListener('click', () => {
            this.addChildNode(this.currentNodeId);
            this.hideContextMenu();
        });
        
        // Add parent node action
        document.getElementById('addParentAction').addEventListener('click', () => {
            if (this.currentNodeId !== 'root') {
                this.addParentNode(this.currentNodeId);
            }
            this.hideContextMenu();
        });
        
        // Focus on branch action
        document.getElementById('focusNodeAction').addEventListener('click', () => {
            this.focusOnNode(this.currentNodeId);
            this.hideContextMenu();
        });
        
        // Expand/collapse all children action
        document.getElementById('expandCollapseAction').addEventListener('click', () => {
            this.toggleExpandCollapseAll(this.currentNodeId);
            this.hideContextMenu();
        });
        
        // Open color picker
        document.getElementById('nodeColorAction').addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleColorPicker();
        });
        
        // Delete node action
        document.getElementById('deleteNodeAction').addEventListener('click', () => {
            this.deleteNode(this.currentNodeId);
            this.hideContextMenu();
        });
        
        // Color picker options
        const colorOptions = document.querySelectorAll('.color-option');
        colorOptions.forEach(option => {
            option.addEventListener('click', (e) => {
                const color = e.target.getAttribute('data-color');
                this.changeNodeColor(this.currentNodeId, color);
                this.hideContextMenu();
            });
        });
    }

    showContextMenu(x, y) {
        // Reset color picker visibility
        this.colorPicker.style.display = 'none';
        
        // Position the context menu
        this.contextMenu.style.left = `${x}px`;
        this.contextMenu.style.top = `${y}px`;
        
        // Show the context menu
        this.contextMenu.classList.add('visible');
    }

    hideContextMenu() {
        // Hide the context menu
        this.contextMenu.classList.remove('visible');
        this.colorPicker.style.display = 'none';
    }
    
    toggleColorPicker() {
        // Toggle color picker visibility
        this.colorPicker.style.display = this.colorPicker.style.display === 'none' ? 'block' : 'none';
    }
    
    editNodeText(nodeId) {
        // Get the node element and its position
        const node = this.mindmap.get_node(nodeId);
        const nodeElement = document.querySelector(`jmnode[nodeid="${nodeId}"]`);
        
        if (!node || !nodeElement) return;
        
        // Get node position
        const rect = nodeElement.getBoundingClientRect();
        
        // Create input element
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'node-edit-input';
        input.value = node.topic;
        
        // Position input over the node
        input.style.left = `${rect.left}px`;
        input.style.top = `${rect.top}px`;
        input.style.width = `${Math.max(rect.width, 150)}px`;
        
        // Add input to DOM
        document.body.appendChild(input);
        
        // Focus and select text
        input.focus();
        input.select();
        
        // Handle completion of editing
        const completeEdit = () => {
            const newText = input.value.trim();
            if (newText && newText !== node.topic) {
                this.mindmap.update_node(nodeId, newText);
            }
            document.body.removeChild(input);
        };
        
        // Event listeners for input
        input.addEventListener('blur', completeEdit);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                completeEdit();
            } else if (e.key === 'Escape') {
                document.body.removeChild(input);
            }
        });
    }
    
    addChildNode(parentId) {
        // Generate a unique ID for the new node
        const newNodeId = `node_${Date.now()}`;
        
        // Add a new child node
        this.mindmap.add_node(parentId, newNodeId, 'New Node');
        
        // Ensure the parent is expanded
        this.mindmap.expand_node(parentId);
        
        // After a small delay, edit the new node
        setTimeout(() => {
            this.editNodeText(newNodeId);
        }, 100);
    }
    
    addParentNode(childId) {
        // This is a more complex operation as jsMind doesn't have direct support
        const child = this.mindmap.get_node(childId);
        if (!child || childId === 'root') return;
        
        const originalParentId = child.parent.id;
        
        // Generate a unique ID for the new parent node
        const newParentId = `node_${Date.now()}`;
        
        // Add the new parent to the original parent
        this.mindmap.add_node(originalParentId, newParentId, 'New Parent');
        
        // Move the child node to be under the new parent
        this.mindmap.move_node(childId, newParentId);
        
        // Ensure nodes are expanded
        this.mindmap.expand_node(originalParentId);
        this.mindmap.expand_node(newParentId);
        
        // After a small delay, edit the new node
        setTimeout(() => {
            this.editNodeText(newParentId);
        }, 100);
    }
    
    focusOnNode(nodeId) {
        // First, ensure the node and all its ancestors are expanded
        let node = this.mindmap.get_node(nodeId);
        while (node && node.id !== 'root') {
            this.mindmap.expand_node(node.parent.id);
            node = node.parent;
        }
        
        // Get the node element
        const nodeElement = document.querySelector(`jmnode[nodeid="${nodeId}"]`);
        if (!nodeElement) return;
        
        // Get node position and center the view on it
        const rect = nodeElement.getBoundingClientRect();
        const containerRect = document.querySelector('.fullscreen-mindmap').getBoundingClientRect();
        
        // Calculate the center position
        const centerX = containerRect.width / 2;
        const centerY = containerRect.height / 2;
        
        // Calculate the target position to center the node
        this.panX = centerX - rect.left - rect.width / 2;
        this.panY = centerY - rect.top - rect.height / 2;
        
        // Apply the transformation
        this.applyTransform();
    }
    
    toggleExpandCollapseAll(nodeId) {
        const node = this.mindmap.get_node(nodeId);
        if (!node) return;
        
        const isExpanded = this.mindmap.get_node_expanded(node);
        
        if (isExpanded) {
            // Collapse all descendants
            this.collapseAllChildren(nodeId);
        } else {
            // Expand all descendants
            this.expandAllChildren(nodeId);
        }
    }
    
    expandAllChildren(nodeId) {
        // Recursively expand all nodes
        const expandRecursively = (id) => {
            const node = this.mindmap.get_node(id);
            if (!node) return;
            
            this.mindmap.expand_node(id);
            
            if (node.children && node.children.length > 0) {
                node.children.forEach(child => {
                    expandRecursively(child.id);
                });
            }
        };
        
        expandRecursively(nodeId);
    }
    
    collapseAllChildren(nodeId) {
        // Recursively collapse all nodes
        const collapseRecursively = (id) => {
            const node = this.mindmap.get_node(id);
            if (!node) return;
            
            if (node.children && node.children.length > 0) {
                node.children.forEach(child => {
                    collapseRecursively(child.id);
                });
                this.mindmap.collapse_node(id);
            }
        };
        
        // We don't want to collapse the starting node itself
        const node = this.mindmap.get_node(nodeId);
        if (node && node.children && node.children.length > 0) {
            node.children.forEach(child => {
                collapseRecursively(child.id);
            });
        }
    }
    
    changeNodeColor(nodeId, color) {
        if (!nodeId || !color) return;
        
        // Get the node element
        const nodeElement = document.querySelector(`jmnode[nodeid="${nodeId}"]`);
        if (!nodeElement) return;
        
        // Remove existing color classes
        const colorClasses = [
            'node-blue', 'node-green', 'node-red', 'node-purple', 
            'node-orange', 'node-yellow', 'node-teal', 'node-gray'
        ];
        
        colorClasses.forEach(cls => {
            nodeElement.classList.remove(cls);
        });
        
        // Add the new color class
        nodeElement.classList.add(`node-${color}`);
        
        // Store the color in the node data (if needed for persistence)
        const node = this.mindmap.get_node(nodeId);
        if (node) {
            node.data = node.data || {};
            node.data.color = color;
        }
    }
    
    deleteNode(nodeId) {
        // Can't delete the root node
        if (nodeId === 'root') return;
        
        // Confirm deletion
        if (confirm('Are you sure you want to delete this node and all its children?')) {
            this.mindmap.remove_node(nodeId);
        }
    }
}

// Custom Video Player
function initCustomVideoPlayer() {
    const videoPlayer = document.getElementById('videoPlayer');
    const videoControls = document.querySelector('.video-controls');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const progressBar = document.querySelector('.video-progress-bar');
    const progress = document.querySelector('.video-progress');
    const currentTimeDisplay = document.getElementById('currentTime');
    const durationDisplay = document.getElementById('duration');
    const muteBtn = document.getElementById('muteBtn');
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    const videoContainer = document.querySelector('.video-preview');
    
    if (!videoPlayer || !videoControls) return;
    
    // Hide default controls (but keep them accessible for mobile/compatibility)
    videoPlayer.controls = false;
    
    // Play/Pause
    playPauseBtn.addEventListener('click', () => {
        if (videoPlayer.paused) {
            videoPlayer.play();
            videoContainer.classList.add('playing');
        } else {
            videoPlayer.pause();
            videoContainer.classList.remove('playing');
        }
    });
    
    // Update progress bar
    videoPlayer.addEventListener('timeupdate', () => {
        const percent = (videoPlayer.currentTime / videoPlayer.duration) * 100;
        progressBar.style.width = `${percent}%`;
        
        // Update time display
        currentTimeDisplay.textContent = formatTime(videoPlayer.currentTime);
    });
    
    // Click on progress bar to seek
    progress.addEventListener('click', (e) => {
        const rect = progress.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        videoPlayer.currentTime = pos * videoPlayer.duration;
    });
    
    // Load metadata (duration, etc.)
    videoPlayer.addEventListener('loadedmetadata', () => {
        durationDisplay.textContent = formatTime(videoPlayer.duration);
    });
    
    // Mute/Unmute
    muteBtn.addEventListener('click', () => {
        videoPlayer.muted = !videoPlayer.muted;
        muteBtn.innerHTML = videoPlayer.muted ? 
            '<i class="fas fa-volume-mute"></i>' : 
            '<i class="fas fa-volume-up"></i>';
    });
    
    // Fullscreen
    fullscreenBtn.addEventListener('click', () => {
        if (videoContainer.requestFullscreen) {
            videoContainer.requestFullscreen();
        } else if (videoContainer.webkitRequestFullscreen) {
            videoContainer.webkitRequestFullscreen();
        } else if (videoContainer.msRequestFullscreen) {
            videoContainer.msRequestFullscreen();
        }
    });
    
    // Update UI on play/pause
    videoPlayer.addEventListener('play', () => {
        videoContainer.classList.add('playing');
    });
    
    videoPlayer.addEventListener('pause', () => {
        videoContainer.classList.remove('playing');
    });
    
    // Format time to MM:SS
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        seconds = Math.floor(seconds % 60);
        return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    }
}

// Upload Particles Animation
function initUploadParticles() {
    const uploadContainer = document.querySelector('.upload-container');
    const particles = document.querySelector('.particles');
    
    if (!uploadContainer || !particles) return;
    
    // Create particles
    for (let i = 0; i < 15; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random position
        const posX = Math.random() * 100;
        particle.style.left = `${posX}%`;
        particle.style.bottom = '0';
        
        // Random size
        const size = Math.random() * 4 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        
        // Random animation delay and duration
        const delay = Math.random() * 2;
        const duration = Math.random() * 2 + 1;
        particle.style.animation = `particleAnimation ${duration}s ${delay}s infinite ease-out`;
        
        // Add to container
        particles.appendChild(particle);
    }
    
    // Show upload progress when uploading
    const fileInput = document.getElementById('fileInput');
    
    uploadContainer.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Drag and drop functionality with improved feedback
    uploadContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadContainer.classList.add('active');
    });
    
    uploadContainer.addEventListener('dragleave', () => {
        uploadContainer.classList.remove('active');
    });
    
    uploadContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadContainer.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
}

// Progress Steps Visualization
function initProgressSteps() {
    const uploadStep = document.getElementById('uploadStep');
    const transcribeStep = document.getElementById('transcribeStep');
    const analyzeStep = document.getElementById('analyzeStep');
    const visualizeStep = document.getElementById('visualizeStep');
    const progressBarFill = document.querySelector('.progress-bar-fill');
    
    if (!uploadStep || !progressBarFill) return;
    
    // Set initial state
    setActiveStep('upload');
    
    // Update steps based on current operation
    fileInput.addEventListener('change', () => {
        setActiveStep('upload');
    });
    
    transcribeButton.addEventListener('click', () => {
        setTimeout(() => setActiveStep('transcribe'), 100);
    });
    
    analyzeButton.addEventListener('click', () => {
        setTimeout(() => setActiveStep('analyze'), 100);
    });
    
    // Redefine the global setActiveStep function with the proper implementation
    window.setActiveStep = function(step) {
        // Skip if elements aren't available yet
        if (!uploadStep || !progressBarFill) return;
        
        // Reset all steps
        uploadStep.className = 'progress-step';
        transcribeStep.className = 'progress-step';
        analyzeStep.className = 'progress-step';
        visualizeStep.className = 'progress-step';
        
        // Set progress level
        let progressWidth = '0%';
        
        switch(step) {
            case 'upload':
                uploadStep.className = 'progress-step active';
                progressWidth = '25%';
                break;
            case 'transcribe':
                uploadStep.className = 'progress-step completed';
                transcribeStep.className = 'progress-step active';
                progressWidth = '50%';
                break;
            case 'analyze':
                uploadStep.className = 'progress-step completed';
                transcribeStep.className = 'progress-step completed';
                analyzeStep.className = 'progress-step active';
                progressWidth = '75%';
                break;
            case 'visualize':
                uploadStep.className = 'progress-step completed';
                transcribeStep.className = 'progress-step completed';
                analyzeStep.className = 'progress-step completed';
                visualizeStep.className = 'progress-step active';
                progressWidth = '100%';
                break;
        }
        
        // Animate progress bar
        progressBarFill.style.width = progressWidth;
    };
}