<div align="left">
  <img src="assets/logo_bg.png" alt="flowify" width="200"/>
</div>

## Overview
Flowify is an intelligent web application that transforms video content into intuitive, visual mind maps. By leveraging advanced NLP techniques and clustering algorithms, Flowify automatically organizes video content into coherent topics and subtopics. Flowify is purely algorithmic, and doesn't require any generative AI.

<img width="1470" alt="Screenshot 2025-03-17 at 8 16 56 PM" src="https://github.com/user-attachments/assets/db5c6664-e840-4bd0-9e4f-c933eb757749" />

## Features

<img width="1468" alt="Screenshot 2025-03-17 at 8 19 03 PM" src="https://github.com/user-attachments/assets/41d324de-e124-4624-90b7-3356f44ddbf2" />

- **Video Processing**: Upload and process videos of any length

<img width="1466" alt="Screenshot 2025-03-17 at 8 21 24 PM" src="https://github.com/user-attachments/assets/1a82ccbd-6a37-4ab7-97fd-dfeb522b4958" />

- **Automatic Transcription**: Convert speech to text with high accuracy
- **Vector embeddings** of transcript segments

<img width="1470" alt="Screenshot 2025-03-17 at 8 19 55 PM" src="https://github.com/user-attachments/assets/0ed2f6c9-4119-4d95-890a-0d29f1c6909a" />

- **Similarity matrix generation** using cosine similarity
- Dynamic **topic identification** through **sliding submatrix window** across diagonal

<img width="1469" alt="Screenshot 2025-03-17 at 8 20 57 PM" src="https://github.com/user-attachments/assets/e4c03ea3-0e91-4f84-87cd-a4dd6fbcd7eb" />

- **Visual Flow Charts**: Generate clear, hierarchical visualizations of content structure

## Technical Architecture
1. **Frontend**: Web interface for video upload and flow chart visualization
2. **Backend Processing Pipeline**:
   - Video transcription model (ran client side)
   - Text segmentation    
   - Vector embedding generation (nltk TfIdVectorizer)
   - Clustering algorithm (sliding submatrix window)
   - Flow chart generation (JSMind)

## Getting Started
`pip install -r requirements.txt`

`python3 app.py`

## Contributing
This project was created during **TartanHacks 2025**. Feel free to contribute!

## Team
**Team Hackintosh**: [Yuvvan Talreja](https://github.com/yuvvantalreja), [Aryan Daga](https://github.com/aryand2006), [Samatva Kasat](https://github.com/samkas125), [Lakshya Gera](https://github.com/lakrage)
