from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from topic_segmenter import TopicSegmenter

app = Flask(__name__)
CORS(app)  # Add CORS support for cross-domain requests
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'filepath': filepath})

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    if 'transcript' not in request.json:
        return jsonify({'error': 'No transcript provided'}), 400

    transcript = request.json['transcript']
#     transcript = '''
# [0] Translator: Joseph Geni
# Reviewer: Morton Bast There are a lot of ways
# the people around us can help improve our lives We don't bump into every neighbor, so a lot of wisdom never gets passed on,. [26] though we do share the same public spaces So over the past few years,
# I've tried ways to share more with my neighbors in public space, using simple tools like
# stickers, stencils and chalk And these projects came
# from questions I had, like:. [41] How much are my neighbors
# paying for their apartments? (Laughter) How can we lend and borrow more things, without knocking on each
# other's doors at a bad time? How can we share more memories
# of our abandoned buildings,. [56] and gain a better understanding
# of our landscape? How can we share more of our hopes
# for our vacant storefronts, so our communities can reflect
# our needs and dreams today? Now, I live in New Orleans, and I am in love with New Orleans. [74] My soul is always soothed
# by the giant live oak trees, shading lovers, drunks and dreamers
# for hundreds of years, and I trust a city that always
# makes way for music I feel like every time someone sneezes, New Orleans has a parade. [90] (Laughter) The city has some of the most
# beautiful architecture in the world, but it also has one of the highest amounts
# of abandoned properties in America I live near this house, and I thought about how I could
# make it a nicer space for my neighborhood,. [105] and I also thought about something
# that changed my life forever In 2009, I lost someone I loved very much Her name was Joan,
# and she was a mother to me And her death was sudden and unexpected And I thought about death a lot. [132] And  this made me feel deep
# gratitude for the time I've had And  brought clarity to the things
# that are meaningful to my life now But I struggle to maintain
# this perspective in my daily life I feel like it's easy to get
# caught up in the day-to-day, and forget what really matters to you. [160] So with help from old and new friends, I turned the side of this abandoned 
# house into a giant chalkboard, and stenciled it with
# a fill-in-the-blank sentence: "Before I die, I want to " So anyone walking by
# can pick up a piece of chalk,. [176] reflect on their life, and share their personal
# aspirations in public space I didn't know what to expect
# from this experiment, but by the next day,
# the wall was entirely filled out, and it kept growing. [191] And I'd like to share a few things
# that people wrote on this wall "Before I die, I want
# to be tried for piracy" (Laughter) "Before I die, I want to straddle
# the International Dateline" "Before I die, I want
# to sing for millions". [218] "Before I die, I want to plant a tree" "Before I die, I want
# to live off the grid" "Before I die, I want
# to hold her one more time" "Before I die, I want
# to be someone's cavalry" "Before I die, I want
# to be completely myself". [249] So this neglected space
# became a constructive one, and people's hopes and dreams
# made me laugh out loud, tear up, and they consoled me
# during my own tough times It's about knowing you're not alone; it's about understanding our neighbors
# in new and enlightening ways;. [267] it's about making space
# for reflection and contemplation, and remembering what really matters
# most to us as we grow and change I made this last year, and started receiving hundreds
# of messages from passionate people who wanted to make a wall
# with their community. [284] So, my civic center colleagues
# and I made a tool kit, and now walls have been made
# in countries around the world, including Kazakhstan, South Africa, Australia,. [299] Argentina, and beyond Together, we've shown how powerful        
# our public spaces can be if we're given the opportunity
# to have a voice, and share more with one another Two of the most valuable things we have. [315] are time, and our relationships
# with other people In our age of increasing distractions, it's more important than ever
# to find ways to maintain perspective, and remember that life
# is brief and tender Death is something that we're
# often discouraged to talk about,. [332] or even think about, but I've realized that preparing for death is one of the most empowering
# things you can do Thinking about death clarifies your life Our shared spaces can better
# reflect what matters to us,. [347] as individuals and as a community, and with more ways to share
# our hopes, fears and stories, the people around us can not only
# help us make better places, they can help us lead better lives Thank you. [362] (Applause) Thank you (Applause)
# '''

    # Initialize and run topic segmentation
    segmenter = TopicSegmenter(
        window_size=4,
        similarity_threshold=0.25,
        context_size=2,
        min_segment_size=3,
        topic_similarity_threshold=0.35
    )

    try:
        segments, topic_mappings, topic_history = segmenter.segment_transcript(transcript)

        # Format results for frontend, preserving timestamps
        results = []
        for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
            # Preserve the original text with timestamps
            results.append({
                'segment_id': i + 1,
                'topic_name': topic_history[topic_id][1],
                'content': segment  # Keep original timestamped text
            })

        return jsonify({
            'success': True,
            'segments': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)