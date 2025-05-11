# Flowify - Audio Transcription & Topic Analysis

![Flowify Demo](https://i.imgur.com/6BEZ7NR.png)

## About

Flowify is an AI-powered tool that transcribes audio/video content and automatically identifies topic segments. It creates interactive mindmaps to help users visualize the structure and flow of conversations, presentations, or lectures.

## Features

- **Audio Transcription**: Powered by OpenAI's Whisper model, providing accurate transcriptions with timestamps
- **Topic Segmentation**: Automatically identifies topic boundaries and creates hierarchical topic structures
- **Interactive Visualization**: Generates mindmaps to visualize the structure of the content
- **Video Integration**: Timestamps in the transcript link directly to the corresponding point in the video

## How to Use

1. **Upload Video**: Upload your video file (supports MP4, WebM, MOV)
2. **Choose Model**: Select the transcription model (options range from tiny to medium)
3. **Transcribe**: Start the transcription process
4. **Analyze**: Generate the topic segmentation and visualization
5. **Explore**: Navigate through the interactive mindmap and click on nodes to jump to the relevant sections in your video

## Models & Technology

- **Transcription**: OpenAI's Whisper (available in various sizes for speed/accuracy tradeoffs)
- **Topic Segmentation**: Custom algorithm combining TF-IDF, cosine similarity, and hierarchical clustering
- **Frontend**: HTML/CSS/JS with a responsive design
- **Backend**: FastAPI for efficient API endpoints

## API Endpoints

- `POST /transcribe`: Transcribe audio from base64-encoded data
- `POST /analyze`: Analyze transcript to extract topics
- `POST /check_model`: Check if a model is available

## Running Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/flowify
cd flowify
pip install -r requirements.txt
uvicorn fastapi_app:app --host 0.0.0.0 --port 7860 --reload
```

## License

MIT

## Credits

Built using Hugging Face's transformers library and OpenAI's Whisper model.
