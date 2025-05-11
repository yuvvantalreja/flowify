from fastapi import FastAPI, Request, HTTPException, Response, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles as StarletteStaticFiles
import os
import base64
import numpy as np
import torch
import time
from transformers import pipeline
import threading
import json
import logging
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from topic_segmenter import TopicSegmenter
from middleware import HTTPSProxyMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flowify", description="Audio transcription and topic segmentation API")

# Add custom middleware for HTTPS handling
app.add_middleware(HTTPSProxyMiddleware)

# Configure templates with HTTPS handling
templates = Jinja2Templates(directory="templates")

# Custom static file handling to ensure proper URL scheme
class SecureStaticFiles(StarletteStaticFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def get_response(self, path, scope):
        # Force scope to use HTTPS when behind proxy
        if "headers" in scope and any(h[0] == b"x-forwarded-proto" and h[1] == b"https" for h in scope["headers"]):
            scope["scheme"] = "https"
        
        response = await super().get_response(path, scope)
        return response

# Mount static files with secure handler
app.mount("/static", SecureStaticFiles(directory="static"), name="static")

# Models and locks
models = {}
model_lock = threading.Lock()

class AudioData(BaseModel):
    audio: str
    model: Optional[str] = "openai/whisper-base"

class TranscriptData(BaseModel):
    transcript: str

class ModelCheck(BaseModel):
    model: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info(f"Serving index.html with base_url={request.base_url}, url={request.url}")
    
    # Ensure template knows the correct scheme
    context = {"request": request}
    
    return templates.TemplateResponse("index.html", context)

def get_model(model_name):
    """Load and cache the model"""
    with model_lock:
        if model_name not in models:
            logging.info(f"Loading model: {model_name}")
            
            hf_model_name = model_name
            if model_name.startswith('Xenova/'):
                base_name = model_name.split('/')[-1] 
                if '.en' in base_name:
                    size = base_name.split('.')[0].replace('whisper-', '')
                    hf_model_name = f"openai/whisper-{size}"
                else:
                    size = base_name.replace('whisper-', '')
                    hf_model_name = f"openai/whisper-{size}"
                
                logging.info(f"Converting Xenova model {model_name} to {hf_model_name}")
            
            try:
                # Load the model using TensorFlow compatibility mode
                import tensorflow as tf
                logging.info(f"TensorFlow version: {tf.__version__}")
                
                # Configure pipeline with correct parameters to avoid conflicts
                models[model_name] = pipeline(
                    "automatic-speech-recognition", 
                    model=hf_model_name,
                    chunk_length_s=30,
                    stride_length_s=5,
                    framework="tf",  # Explicitly use TensorFlow
                    model_kwargs={
                        "attention_mask": True,  # Explicitly set attention mask
                        "use_cache": True,
                    }
                )
                logging.info(f"Model {model_name} loaded successfully with TensorFlow")
            except Exception as e:
                logging.error(f"Error loading model with TensorFlow: {str(e)}")
                try:
                    # Fallback to PyTorch with more specific configurations
                    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                    
                    # First load processor and model separately to customize configs
                    processor = AutoProcessor.from_pretrained(hf_model_name)
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        hf_model_name,
                        use_cache=True,
                        attention_mask=True
                    )
                    
                    # Create the pipeline with the initialized model and processor
                    models[model_name] = pipeline(
                        "automatic-speech-recognition", 
                        model=model,
                        tokenizer=processor,
                        feature_extractor=processor,
                        chunk_length_s=30,
                        stride_length_s=5,
                        generate_kwargs={
                            "task": "transcribe",
                            "language": "english",
                            # Don't set forced_decoder_ids here to avoid conflict
                        }
                    )
                    logging.info(f"Model {model_name} loaded successfully with PyTorch")
                except Exception as e2:
                    logging.error(f"Failed to load model {model_name}: {str(e2)}")
                    raise RuntimeError(f"Failed to load model {model_name}: {str(e2)}")
        return models[model_name]

def process_audio(audio_data, sample_rate=16000, model_name="openai/whisper-base"):
    """Process audio data with the Whisper model"""
    try:
        model = get_model(model_name)
        
        chunk_duration = 30
        samples_per_chunk = chunk_duration * sample_rate
        total_chunks = (len(audio_data) + samples_per_chunk - 1) // samples_per_chunk
        
        full_transcript = ""
        
        for i in range(total_chunks):
            start = i * samples_per_chunk
            end = min(start + samples_per_chunk, len(audio_data))
            chunk = audio_data[start:end]
            
            logging.info(f"Processing chunk {i+1}/{total_chunks}")
            
            # Use consistent parameters for model inference
            result = model(
                chunk,
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "english"
                }
            )
            
            if "chunks" in result and len(result["chunks"]) > 0:
                for chunk_with_time in result["chunks"]:
                    if "timestamp" in chunk_with_time and len(chunk_with_time["timestamp"]) == 2:
                        chunk_start_seconds = chunk_with_time["timestamp"][0]
                        absolute_start_time = (start / sample_rate) + chunk_start_seconds
                        timestamp = f"[{int(absolute_start_time)}]"
                        full_transcript += f"{timestamp} {chunk_with_time['text'].strip()}\n"
            else:
                timestamp = f"[{int(start / sample_rate)}]"
                full_transcript += f"{timestamp} {result['text'].strip()}\n"
                
        return full_transcript
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        raise

@app.post("/transcribe")
async def transcribe(data: AudioData):
    try:
        audio_base64 = data.audio
        audio_bytes = base64.b64decode(audio_base64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        
        model_name = data.model
        
        start_time = time.time()
        transcript = process_audio(audio_np, model_name=model_name)
        elapsed_time = time.time() - start_time
        
        logging.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        return {"transcript": transcript}
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_topics(data: TranscriptData):
    try:
        transcript = data.transcript
        
        segmenter = TopicSegmenter(
            window_size=2,  
            similarity_threshold=0.15,  
            context_size=1,
            min_segment_size=2,  
            topic_similarity_threshold=0.25,  
            max_topics=6,  
            hierarchical_threshold=0.6  
        )
        
        segments, topic_mappings, topic_history, topic_hierarchies = segmenter.segment_transcript(transcript)
        
        results = []
        
        parent_topics = set()
        for parent_id in topic_hierarchies.keys():
            parent_topics.add(parent_id)
        
        added_segments = set()
        
        for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
            if i in added_segments:
                continue
                
            if topic_id in parent_topics:
                parent_data = {
                    'segment_id': i,
                    'topic_name': topic_history[topic_id][1],
                    'content': segment,
                    'timestamp': topic_history[topic_id][3],
                    'is_parent': True,
                    'children': []
                }
                
                for j, (child_segment, child_topic_id) in enumerate(zip(segments, topic_mappings)):
                    if j != i and child_topic_id in topic_hierarchies.get(topic_id, []):
                        parent_data['children'].append({
                            'segment_id': j,
                            'topic_name': topic_history[child_topic_id][1],
                            'content': child_segment,
                            'timestamp': topic_history[child_topic_id][3]
                        })
                        added_segments.add(j)
                
                results.append(parent_data)
                added_segments.add(i)
            elif topic_id not in [child for children in topic_hierarchies.values() for child in children]:
                results.append({
                    'segment_id': i,
                    'topic_name': topic_history[topic_id][1],
                    'content': segment,
                    'timestamp': topic_history[topic_id][3],
                    'is_parent': False,
                    'children': []
                })
                added_segments.add(i)
        
        return {"segments": results}
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_model")
async def check_model(data: ModelCheck):
    try:
        model_name = data.model
        logging.info(f"Checking model availability: {model_name}")
        
        if model_name in models:
            logging.info(f"Model {model_name} is already loaded")
            return {"available": True}
            
        supported_prefixes = [
            "openai/whisper",
            "Xenova/whisper",
            "whisper"
        ]
        
        available = any(model_name.startswith(prefix) or model_name.lower().startswith(prefix.lower()) for prefix in supported_prefixes)
        
        logging.info(f"Model {model_name} availability: {available}")
        return {"available": available}
    except Exception as e:
        logging.error(f"Model check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Special route for serving the index.html file
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=7860, reload=True) 