# Flowify - Audio Transcription & Topic Analysis

Flowify is an AI-powered tool that transcribes audio/video content and automatically identifies topic segments. It creates interactive mindmaps to help users visualize the structure and flow of conversations, presentations, or lectures.

## Demo

Upload a video file, select a transcription model, and click "Transcribe" to start the process. After transcription, click "Analyze Topics" to generate a topic map.

## Features

- Automatic transcription with OpenAI's Whisper models
- Topic segmentation with hierarchical structure
- Interactive mindmap visualization
- Direct links between transcript timestamps and video playback

## API

This Spaces app also provides API endpoints:

- `POST /transcribe`: Transcribe audio from base64-encoded data
- `POST /analyze`: Analyze transcript to extract topics
- `POST /check_model`: Check if a model is available

## Technical Details

Built with FastAPI and Hugging Face's Transformers library.

## License

MIT 