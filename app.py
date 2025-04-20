from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import base64
import numpy as np
import torch
import time
from transformers import pipeline
import threading
import json
import logging
from topic_segmenter import TopicSegmenter

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Cache for loaded models
models = {}
model_lock = threading.Lock()

# Serve static files
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def get_model(model_name):
    """Load and cache the model"""
    with model_lock:
        if model_name not in models:
            logging.info(f"Loading model: {model_name}")
            

            hf_model_name = model_name
            if model_name.startswith('Xenova/'):

                base_name = model_name.split('/')[-1] 
                if '.en' in base_name:
                    # Handle English-only models
                    size = base_name.split('.')[0].replace('whisper-', '')
                    hf_model_name = f"openai/whisper-{size}"
                else:
                    size = base_name.replace('whisper-', '')
                    hf_model_name = f"openai/whisper-{size}"
                
                logging.info(f"Converting Xenova model {model_name} to {hf_model_name}")
            
            models[model_name] = pipeline(
                "automatic-speech-recognition", 
                hf_model_name,
                chunk_length_s=30,
                stride_length_s=5
            )
            logging.info(f"Model {model_name} loaded successfully")
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
            
            result = model(
                chunk,
                return_timestamps=True,
                generate_kwargs={"language": "english", "task": "transcribe"}
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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.json
        
        if not data or 'audio' not in data:
            return jsonify({"error": "No audio data provided"}), 400
            
        # Extract audio data
        audio_base64 = data['audio']
        audio_bytes = base64.b64decode(audio_base64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Get model name or use default
        model_name = data.get('model', 'openai/whisper-base')
        
        # Process audio in a separate thread
        start_time = time.time()
        transcript = process_audio(audio_np, model_name=model_name)
        elapsed_time = time.time() - start_time
        
        logging.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        return jsonify({"transcript": transcript})
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_topics():
    try:
        data = request.json
        
        if not data or 'transcript' not in data:
            return jsonify({"error": "No transcript provided"}), 400
            
        transcript = data['transcript']
        
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
        
        return jsonify({"segments": results})
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_model', methods=['POST'])
def check_model():
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({"error": "No model specified"}), 400
            
        model_name = data['model']
        logging.info(f"Checking model availability: {model_name}")
        
        if model_name in models:
            logging.info(f"Model {model_name} is already loaded")
            return jsonify({"available": True})
            
        supported_prefixes = [
            "openai/whisper",
            "Xenova/whisper",
            "whisper"
        ]
        
        available = any(model_name.startswith(prefix) or model_name.lower().startswith(prefix.lower()) for prefix in supported_prefixes)
        
        logging.info(f"Model {model_name} availability: {available}")
        return jsonify({"available": available})
    except Exception as e:
        logging.error(f"Model check error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)