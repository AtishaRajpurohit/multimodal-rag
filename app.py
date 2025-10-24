"""
Multimodal RAG Streamlit Application
===================================

A modern web interface for face detection, matching, and multimodal image description.
Integrates with the main.py pipeline for end-to-end processing.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from loguru import logger

# Import our main pipeline
from src.main import process_image_pipeline

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Multimodal RAG - Face Detection & Description",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Mode selection buttons */
    .mode-button {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .mode-button:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    .mode-button.selected {
        border-color: #667eea;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Output box styling */
    .output-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Status messages */
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("## Configuration")
    
    # Collection name input
    collection_name = st.text_input(
        "Qdrant Collection Name",
        value="reference_dataset_collection",
        help="Name of the Qdrant collection to search for face matches"
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("## System Status")
    
    # Check if Qdrant is running
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        st.success("Qdrant: Connected")
        st.info(f"Collections: {len(collections.collections)}")
    except Exception as e:
        st.error("Qdrant: Not Connected")
        st.warning("Please ensure Qdrant is running on localhost:6333")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("## Instructions")
    st.markdown("""
    1. **Take Photo**: Use the camera to capture an image
    2. **Retake**: If needed, retake the photo
    3. **Select Mode**: Choose description style
    4. **Process**: Run the face detection and description pipeline
    5. **View Results**: See detected faces and generated description
    """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown('<h1 class="main-title">Multimodal RAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Face Detection, Matching & AI-Powered Image Description</p>', unsafe_allow_html=True)

# Initialize session state
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = None

# =============================================================================
# CAMERA SECTION
# =============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## 📸 Capture Image")

# Camera input
img_file = st.camera_input("Take a picture for face detection and description")

if img_file is not None:
    # Save the image temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(img_file.getbuffer())
            temp_image_path = tmp_file.name
        
        st.session_state.captured_image = temp_image_path
        st.session_state.image_captured = True
        
        # Display the captured image
        st.image(img_file, caption="Captured Image", use_column_width=True)
        
        st.markdown('<div class="status-success">✅ Image captured successfully!</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="status-error">❌ Error saving image: {str(e)}</div>', unsafe_allow_html=True)
        st.session_state.image_captured = False

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MODE SELECTION SECTION
# =============================================================================
if st.session_state.image_captured:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🎨 Select Description Mode")
    
    # Mode selection buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Humanlike", key="mode_humanlike", use_container_width=True):
            st.session_state.selected_mode = "humanlike"
    
    with col2:
        if st.button("Detailed", key="mode_detailed", use_container_width=True):
            st.session_state.selected_mode = "detailed"
    
    with col3:
        if st.button("Funny", key="mode_funny", use_container_width=True):
            st.session_state.selected_mode = "funny"
    
    # Display selected mode
    if st.session_state.selected_mode:
        mode_descriptions = {
            "humanlike": "Natural, conversational description of the scene",
            "detailed": "Comprehensive analysis of clothing, poses, and relationships",
            "funny": "Humorous and light-hearted interpretation"
        }
        
        st.markdown(f"""
        <div class="status-success">
            <strong>Selected Mode:</strong> {st.session_state.selected_mode.title()}<br>
            <em>{mode_descriptions[st.session_state.selected_mode]}</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PROCESSING SECTION
# =============================================================================
if st.session_state.image_captured and st.session_state.selected_mode:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## ⚡ Process Image")
    
    if st.button("🚀 Run Face Detection & Description", use_container_width=True):
        if st.session_state.captured_image:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                progress_bar.progress(25)
                status_text.text("Initializing components...")
                
                # Run the pipeline
                progress_bar.progress(50)
                status_text.text("Detecting faces and searching matches...")
                
                result = process_image_pipeline(
                    image_path=st.session_state.captured_image,
                    collection_name=collection_name,
                    description_mode=st.session_state.selected_mode
                )
                
                progress_bar.progress(75)
                status_text.text("Generating description...")
                
                # Update progress
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Store results in session state
                st.session_state.processing_result = result
                
                # Status update
                st.write("Processing completed")
                st.write("Result keys:", list(result.keys()) if result else "No result")
                if result and 'description' in result:
                    st.write("Description found, length:", len(result['description']))
                    st.write("Description preview:", result['description'][:300] + "..." if len(result['description']) > 300 else result['description'])
                else:
                    st.write("No description in result")
                
            except Exception as e:
                st.markdown(f'<div class="status-error">❌ Error processing image: {str(e)}</div>', unsafe_allow_html=True)
                logger.error(f"Streamlit processing error: {e}")
                st.write("Full error details:", str(e))
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# RESULTS SECTION
# =============================================================================
if 'processing_result' in st.session_state and st.session_state.processing_result:
    result = st.session_state.processing_result
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📊 Results")
    
    if result['success']:
        # Face detection results
        st.markdown("### 👥 Detected Faces")
        
        if result['faces']:
            for i, face in enumerate(result['faces']):
                match_info = face.get('match', {'label': 'Unknown', 'score': 'N/A'})
                confidence = face.get('face_confidence', 0)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.metric("Face", f"{i+1}")
                with col2:
                    st.metric("Person", match_info['label'])
                with col3:
                    st.metric("Confidence", f"{confidence:.3f}")
                
                if match_info['score'] != 'N/A':
                    st.metric("Match Score", f"{match_info['score']:.3f}")
                
                st.markdown("---")
        else:
            st.warning("No faces detected in the image")
        
        # Description results
        st.markdown("### 📝 Generated Description")
        
        # Display the description
        description = result.get('description', '')
        if description and description.strip():
            st.markdown(f"""
            <div class="output-box">
                <strong>Mode:</strong> {st.session_state.selected_mode.title()}<br><br>
                {description}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-error">
                ❌ No description generated. This might be due to:
                <ul>
                    <li>OpenAI API key not set</li>
                    <li>Network connection issues</li>
                    <li>API rate limits</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Show debug info
            st.write("Debug - Description field:", repr(description))
            st.write("Debug - Description length:", len(description))
            st.write("Debug - Full result keys:", list(result.keys()))
        
    else:
        st.markdown(f'<div class="status-error">❌ Processing failed: {result["description"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Multimodal RAG Application | Powered by DeepFace, Qdrant & OpenAI GPT-4o</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CLEANUP
# =============================================================================
# Clean up temporary files when session ends
if 'captured_image' in st.session_state and st.session_state.captured_image:
    try:
        if os.path.exists(st.session_state.captured_image):
            os.unlink(st.session_state.captured_image)
    except Exception as e:
        logger.warning(f"Could not clean up temporary file: {e}")