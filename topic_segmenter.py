import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
import re
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

class TopicSegmenter:
    def __init__(self, window_size=3, similarity_threshold=0.2, context_size=2, min_segment_size=2, topic_similarity_threshold=0.3, 
                max_topics=8, hierarchical_threshold=0.6, model_name="all-MiniLM-L6-v2"):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.context_size = context_size
        self.min_segment_size = min_segment_size
        self.topic_similarity_threshold = topic_similarity_threshold
        self.max_topics = max_topics  # Maximum number of main topics
        self.hierarchical_threshold = hierarchical_threshold  # Threshold for hierarchical clustering
        
        try:
            # Explicitly set the model to a lightweight and compatible one
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent parallelism warning
            
            # Try using sentence-transformers directly first
            try:
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer(model_name)
                self.keyword_extractor = KeyBERT(model=embedding_model)
            except (ImportError, ValueError, RuntimeError) as e:
                # Fallback to default model with lower-level settings
                self.keyword_extractor = KeyBERT(model='distilbert-base-nli-mean-tokens')
        except Exception as e:
            import logging
            logging.error(f"Error initializing KeyBERT model: {str(e)}")
            # Last resort fallback to simple keyword extraction without a model
            self.keyword_extractor = None
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        self.segment_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 2)
        )
        
        self.lemmatizer = WordNetLemmatizer()
        self.topic_history = []
        self.topic_hierarchy = {}  # Store parent-child relationships
        self.fitted_vectorizer = None
        
    def preprocess_text(self, text):
        # First, split the text by timestamps
        timestamp_chunks = re.findall(r'\[(\d+)\](.*?)(?=\[\d+\]|$)', text, re.DOTALL)
        
        all_sentences = []
        original_sentences = []
        timestamp_mapping = []  # To track which timestamp each sentence belongs to
        
        for timestamp, chunk_text in timestamp_chunks:
            # Split each chunk into sentences
            chunk_sentences = sent_tokenize(chunk_text.strip())
            for sentence in chunk_sentences:
                if sentence.strip():
                    # Keep track of the original sentence
                    original_sentences.append(sentence.strip())
                    
                    # Clean and process the sentence for analysis
                    clean_sentence = re.sub(r'^[^:]+:', '', sentence).strip()
                    clean_sentence = re.sub(r'[^\w\s]', '', clean_sentence.lower())
                    words = word_tokenize(clean_sentence)
                    lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
                    processed_sentence = ' '.join(lemmatized)
                    
                    all_sentences.append(processed_sentence)
                    timestamp_mapping.append(int(timestamp))
        
        return all_sentences, original_sentences, timestamp_mapping
    
    def get_topic_fingerprint(self, segment_text):
        cleaned_text = ' '.join([re.sub(r'^[^:]+:', '', sent).strip() for sent in segment_text])
        
        if self.fitted_vectorizer is None:
            self.fitted_vectorizer = self.segment_vectorizer.fit([cleaned_text])
            tfidf_matrix = self.fitted_vectorizer.transform([cleaned_text])
        else:
            tfidf_matrix = self.fitted_vectorizer.transform([cleaned_text])
            
        return tfidf_matrix.toarray()[0]

    def compare_with_previous_topics(self, current_segment):
        if not self.topic_history:
            return None, 0.0

        current_fingerprint = self.get_topic_fingerprint(current_segment)
        max_similarity = 0.0
        best_match_idx = None

        for idx, (topic_fingerprint, _, _, _, _) in enumerate(self.topic_history):
            if len(topic_fingerprint) != len(current_fingerprint):
                continue
                
            similarity = cosine_similarity([topic_fingerprint], [current_fingerprint])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_idx = idx

        return best_match_idx, max_similarity
    
    def extract_keywords(self, sentences, top_n=3):
        if isinstance(sentences, list):
            text = ' '.join(sentences)
        else:
            text = sentences
        
        # If KeyBERT was not properly initialized, use TF-IDF as a fallback
        if self.keyword_extractor is None:
            try:
                # Use scikit-learn's TF-IDF for keyword extraction as fallback
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Create a temporary vectorizer for this text only
                temp_vectorizer = TfidfVectorizer(
                    max_features=50,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                # Fit and transform on this specific text
                tfidf_matrix = temp_vectorizer.fit_transform([text])
                feature_names = temp_vectorizer.get_feature_names_out()
                
                # Get top keywords based on TF-IDF scores
                tfidf_scores = tfidf_matrix.toarray()[0]
                scored_tokens = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
                sorted_tokens = sorted(scored_tokens, key=lambda x: x[1], reverse=True)
                
                return [token for token, score in sorted_tokens[:top_n]]
            except Exception as e:
                import logging
                logging.error(f"Error in TF-IDF fallback keyword extraction: {str(e)}")
                # If everything fails, just return some generic keywords
                return ["topic", "section", "content"]
        
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2),  # Allow for 1-2 word keyphrases
                stop_words='english',
                top_n=top_n
            )
            
            sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
            return [kw[0] for kw in sorted_keywords[:top_n]]
        except Exception as e:
            import logging
            logging.error(f"Error in KeyBERT keyword extraction: {str(e)}")
            # Fall back to TF-IDF method
            return self.extract_keywords(text, top_n)

    def detect_speaker_changes(self, text):
        """
        Detect points in the transcript where speakers change.
        Returns a list of indices where speaker changes occur.
        """
        speaker_pattern = re.compile(r'(?:^|\s)(?:I\'m|And I\'m|Hey I\'m|I am)\s+([A-Z][a-z]+)', re.MULTILINE)
        matches = list(speaker_pattern.finditer(text))
        
        speaker_boundaries = []
        for match in matches:
            # Find the sentence containing this speaker introduction
            sentences = sent_tokenize(text[:match.end()])
            if sentences:
                speaker_boundaries.append(len(sentences) - 1)
        
        return speaker_boundaries

    def find_closest_timestamp(self, sentence, transcript):
        """
        Find the timestamp after the most recent timestamp before the sentence.
        """
        chunks = re.findall(r'\[(\d+)\](.*?)(?=\[\d+\]|$)', transcript, re.DOTALL)
        sentence = sentence.strip().lower()
        
        previous_timestamp = 0
        current_timestamp = 0
        
        for i, (timestamp, text) in enumerate(chunks):
            current_timestamp = int(timestamp)
            text = text.strip().lower()
            
            if sentence in text or text in sentence:
                if i > 0 and i < len(chunks):
                    return current_timestamp
                return previous_timestamp
            
            previous_timestamp = current_timestamp
                
        return current_timestamp 

    def calculate_topic_importance(self, segment, total_segments):
        """
        Calculate importance score for a topic based on:
        1. Length of the segment
        2. Position in the transcript
        3. Keyword significance
        """
        # Length factor - longer segments likely more important
        length_score = min(1.0, len(segment) / 10)  # Cap at 1.0 for segments of 10+ sentences
        
        # Position factor - beginning and end often contain important information
        position_in_doc = self.topic_history.index(segment) / total_segments
        position_score = 1.0 - min(abs(position_in_doc - 0.0), abs(position_in_doc - 1.0))
        
        # Keyword significance - look for key terms indicating importance
        importance_indicators = ['important', 'key', 'main', 'primary', 'critical', 'essential', 
                                'significant', 'revolutionary', 'breakthrough', 'innovative']
        
        text = ' '.join(segment)
        significance_score = 0.0
        for indicator in importance_indicators:
            if indicator in text.lower():
                significance_score += 0.2  # Add 0.2 for each indicator found
        significance_score = min(1.0, significance_score)  # Cap at 1.0
        
        # Combined score, weighted
        combined_score = (0.4 * length_score) + (0.3 * position_score) + (0.3 * significance_score)
        return combined_score
    
    def cluster_topics(self, segments):
        """
        Cluster similar topics together to form a hierarchy
        """
        if len(segments) <= 1:
            return segments, []
            
        # Extract features for each segment
        feature_vectors = []
        for segment_idx, segment in enumerate(segments):
            feature_vectors.append(self.get_topic_fingerprint(segment))
            
        feature_matrix = np.array(feature_vectors)
        
        # Use hierarchical clustering with compatible parameters
        num_clusters = min(self.max_topics, len(segments))
        
        # Try different parameter combinations based on scikit-learn version compatibility
        try:
            # First attempt with basic parameters
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters
            )
            labels = clustering.fit_predict(feature_matrix)
        except Exception as e:
            print(f"Clustering with default parameters failed: {str(e)}")
            # Fallback to manual clustering based on similarity
            labels = self.manual_clustering(feature_matrix, num_clusters)
        
        # Group segments by cluster
        clusters = defaultdict(list)
        for segment_idx, cluster_id in enumerate(labels):
            clusters[cluster_id].append(segment_idx)
            
        # Create parent-child relationships
        parent_child_map = {}
        
        for cluster_id, segment_indices in clusters.items():
            # If cluster has multiple segments, create a parent topic
            if len(segment_indices) > 1:
                # Combine segments to create a parent topic
                all_content = []
                for idx in segment_indices:
                    all_content.extend(segments[idx])
                
                # Extract the most representative keywords
                keywords = self.extract_keywords(all_content, top_n=3)
                parent_topic = f"{', '.join(keywords)}"
                
                # Add parent-child relationships
                for idx in segment_indices:
                    parent_child_map[idx] = parent_topic
        
        return labels, parent_child_map
    
    def manual_clustering(self, feature_matrix, num_clusters):
        """
        Implement a simple similarity-based clustering as fallback
        """
        n_samples = feature_matrix.shape[0]
        
        # If we have very few samples, each gets its own cluster
        if n_samples <= num_clusters:
            return np.arange(n_samples)
            
        # Calculate similarity matrix
        similarity = cosine_similarity(feature_matrix)
        
        labels = np.arange(n_samples)
        current_num_clusters = n_samples
        
        while current_num_clusters > num_clusters:
            max_similarity = -1
            merge_i, merge_j = 0, 0         
            
            for i in range(n_samples):
                for j in range(i+1, n_samples):

                    if labels[i] == labels[j]:
                        continue
                        
                    if similarity[i, j] > max_similarity:
                        max_similarity = similarity[i, j]
                        merge_i, merge_j = i, j
            
            old_label = labels[merge_j]
            new_label = labels[merge_i]
            
            for i in range(n_samples):
                if labels[i] == old_label:
                    labels[i] = new_label
            
            unique_labels = np.unique(labels)
            mapping = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([mapping[label] for label in labels])
            
            current_num_clusters -= 1
            
        return labels
    
    def segment_transcript(self, text):
        cleaned_sentences, original_sentences, timestamp_mapping = self.preprocess_text(text)
        
        if len(cleaned_sentences) == 0:
            return [], [], [], {}
        
        similarity_matrix = self.calculate_similarity_matrix(cleaned_sentences)
        
        content_boundaries = self.detect_topic_boundaries(similarity_matrix)
        
        speaker_boundaries = self.detect_speaker_changes(text)
        
        timestamp_boundaries = []
        for i in range(1, len(timestamp_mapping)):
            if timestamp_mapping[i] - timestamp_mapping[i-1] > 5:  # If there's a gap of more than 5 seconds
                timestamp_boundaries.append(i)
        
        all_boundaries = sorted(set(content_boundaries + speaker_boundaries + timestamp_boundaries))
        
        initial_segments = []
        initial_segment_texts = []
        current_topic_id = 0
        self.topic_history = []  # Reset topic history
        
        start_idx = 0
        for boundary in all_boundaries + [len(original_sentences)]:
            if boundary - start_idx < self.min_segment_size:
                continue
                
            current_segment = original_sentences[start_idx:boundary]
            segment_timestamps = timestamp_mapping[start_idx:boundary]
            
            closest_timestamp = segment_timestamps[0] if segment_timestamps else 0
            
            # Extract keywords directly from content without predefined categories
            keywords = self.extract_keywords(current_segment, top_n=4)
            
            # Create topic name directly from the top keywords - completely dynamic
            topic_name = f"{', '.join(keywords[:2])}"
            
            # Calculate importance score
            topic_importance = len(current_segment) / 3  # Simple approach: longer segments are more important
            
            # Create a topic fingerprint
            topic_fingerprint = self.get_topic_fingerprint(current_segment)
            
            # Store topic information
            self.topic_history.append((topic_fingerprint, topic_name, [current_segment], closest_timestamp, topic_importance))
            initial_segments.append(current_segment)
            initial_segment_texts.append(topic_name)
            
            start_idx = boundary
        
        # Cluster segments to form hierarchical topics
        if len(initial_segments) > 1:
            cluster_labels, parent_child_map = self.cluster_topics(initial_segments)
        else:
            cluster_labels = [0] if initial_segments else []
            parent_child_map = {}
        
        # Organize segments into hierarchical structure
        final_segments = []
        topic_mappings = []
        topic_hierarchies = {}
        
        # Group by parent topics
        parent_topic_groups = defaultdict(list)
        
        for i, segment in enumerate(initial_segments):
            topic_id = i
            
            # Check if this segment belongs to a parent topic
            if i in parent_child_map:
                parent_topic = parent_child_map[i]
                parent_id = len(self.topic_history)
                
                # If this is the first time seeing this parent topic
                if parent_topic not in parent_topic_groups:
                    # Add the parent topic to our history
                    all_content = []
                    for j, seg in enumerate(initial_segments):
                        if j in parent_child_map and parent_child_map[j] == parent_topic:
                            all_content.extend(seg)
                    
                    parent_fingerprint = self.get_topic_fingerprint(all_content)
                    parent_timestamp = self.topic_history[i][3]  # Use timestamp of first segment
                    parent_importance = 1.0  # Parent topics are most important
                    
                    self.topic_history.append((parent_fingerprint, parent_topic, [all_content], parent_timestamp, parent_importance))
                    topic_hierarchies[parent_id] = [j for j, seg in enumerate(initial_segments) 
                                               if j in parent_child_map and parent_child_map[j] == parent_topic]
                
                # Find the parent ID
                for j, (_, name, _, _, _) in enumerate(self.topic_history):
                    if name == parent_topic:
                        parent_id = j
                        break
                
                # Add to parent topic group
                parent_topic_groups[parent_topic].append((segment, topic_id))
            else:
                # This is already a main topic
                final_segments.append(segment)
                topic_mappings.append(topic_id)
        
        # Add grouped segments under their parent topics
        for parent_topic, segments_and_ids in parent_topic_groups.items():
            # Find the parent ID
            parent_id = None
            for i, (_, name, _, _, _) in enumerate(self.topic_history):
                if name == parent_topic:
                    parent_id = i
                    break
            
            if parent_id is not None:
                # Add parent as a segment
                parent_content = []
                for segment, _ in segments_and_ids:
                    parent_content.extend(segment)
                
                final_segments.append(parent_content)
                topic_mappings.append(parent_id)
                
                # Add children segments
                for segment, child_id in segments_and_ids:
                    final_segments.append(segment)
                    topic_mappings.append(child_id)
        
        # For topics that don't have many segments, merge them if they're similar
        if len(final_segments) > self.max_topics * 2:
            merged_segments = []
            merged_mappings = []
            i = 0
            
            while i < len(final_segments):
                current_segment = final_segments[i]
                current_mapping = topic_mappings[i]
                
                # If this is a short segment, try to merge with next
                if len(current_segment) <= 2 and i + 1 < len(final_segments):
                    next_segment = final_segments[i+1]
                    combined = current_segment + next_segment
                    
                    # Use keywords to see if they're related
                    current_keywords = self.extract_keywords(current_segment)
                    next_keywords = self.extract_keywords(next_segment)
                    
                    # Check for keyword overlap
                    overlap = any(kw in next_keywords for kw in current_keywords)
                    
                    if overlap:
                        # Merge the segments
                        merged_segments.append(combined)
                        merged_mappings.append(current_mapping)  # Keep the first mapping
                        i += 2  # Skip both segments
                        continue
                
                # If no merge, keep as is
                merged_segments.append(current_segment)
                merged_mappings.append(current_mapping)
                i += 1
            
            final_segments = merged_segments
            topic_mappings = merged_mappings
            
        return final_segments, topic_mappings, self.topic_history, topic_hierarchies

    def calculate_similarity_matrix(self, sentences):
        if not sentences:
            return np.array([[]])
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        return cosine_similarity(tfidf_matrix)

    def detect_topic_boundaries(self, similarity_matrix):
        boundaries = []
        n_sentences = len(similarity_matrix)
        
        if n_sentences <= 2 * self.window_size:
            return []
        
        for i in range(self.window_size, n_sentences - self.window_size):
            prev_window = similarity_matrix[i-self.window_size:i, i-self.window_size:i]
            prev_similarity = np.mean(prev_window)
            
            next_window = similarity_matrix[i:i+self.window_size, i:i+self.window_size]
            next_similarity = np.mean(next_window)
            
            cross_window = similarity_matrix[i-self.window_size:i, i:i+self.window_size]
            cross_similarity = np.mean(cross_window)
            
            # Stricter conditions for detecting boundaries
            if (cross_similarity < self.similarity_threshold and
                cross_similarity < prev_similarity * 0.7 and  # Increased constraint
                cross_similarity < next_similarity * 0.7 and  # Increased constraint
                (len(boundaries) == 0 or i - boundaries[-1] >= self.min_segment_size)):
                boundaries.append(i)
        
        return boundaries
    
if __name__ == "__main__":
    transcript = """
[0] Hey I'm Stanley, I'm a Stanford CS major. I did fun at Engineering at Facebook. [4] Hey I'm Andy, I'm also a Stanford CS major and I did platform engineering at Facebook. [8] I'm Evan, I was on the founding team of Vivo, the music video service. [12] And I'm Tony, I was a product editor at Square. [15] And the four of us came together about six months ago to work on software for small business owners. [20] But we didn't have a need at first so we just went out and talked to all the small business owners we could find. [24] [6] After over a hundred interviews, we came across a really interesting problem with small business restaurants in an area like this. [30] Yeah, so it turns out restaurants in Palo Alto don't deliver even though they really [34] want to, but they can't afford it. [36] But their consumers are craving for it, but the places that the consumers love just can't [41] deliver. [42] And we also found out about these delivery drivers who had a ton of spare time and they [45] all wanted an extra cash during that downtime. [48] Right, so that's when we built an initial product, PaloAltodelivery.com, and how it [53] worked is the customer goes to the website and places it all in order that gets automatically [59] [32] since the restaurant [60] And then we as the dispatchers with some pretty neat routing and badging algorithms were able to send the drivers efficiently to [66] Get the orders to the customers at a really fast time. Yeah, and the four of us actually started off as delivery drivers and [74] Over time we hired more as we grew and in our first month of launch with not much marketing [80] We got over 150 paying customers in the Palo Alto area, which was really awesome [84] And from that we generated over $10,000 in sales
    """
    
    segmenter = TopicSegmenter(
        window_size=2,  # Smaller window for more granular segmentation
        similarity_threshold=0.15,  # Lower threshold to detect more subtle changes
        context_size=1,
        min_segment_size=2,  # Smaller minimum size for segments
        topic_similarity_threshold=0.25,  # Lower threshold to better distinguish topics
        max_topics=5,  # Target number of main topics
        hierarchical_threshold=0.6  # Threshold for hierarchical clustering
    )
    
    segments, topic_mappings, topic_history, topic_hierarchies = segmenter.segment_transcript(transcript)
    
    print("Topic Segmentation Analysis:\n")
    print(f"Found {len(set(topic_mappings))} distinct topics")
    
    # Print hierarchical topic structure
    print("\nTopic Hierarchy:")
    for parent_id, child_ids in topic_hierarchies.items():
        parent_name = topic_history[parent_id][1]
        print(f"Main Topic: {parent_name}")
        for child_id in child_ids:
            child_name = topic_history[child_id][1]
            print(f"  - Subtopic: {child_name}")
    
    print("\nDetailed Segments:")
    for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
        print(f"Segment {i+1} (Part of {topic_history[topic_id][1]}):")
        print(f"Closest Timestamp: {topic_history[topic_id][3]} seconds")
        print("-" * 50)
        print("\n".join(segment[:2]) + "..." if len(segment) > 2 else "\n".join(segment))
        print("-" * 50 + "\n")