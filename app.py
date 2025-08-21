import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from urllib.parse import urlparse, urljoin
import hashlib
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ---------------------------
# ✅ Page config
# ---------------------------
st.set_page_config(
    page_title="🧠 Advanced Semantic Similarity Checker v3.0", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# ✅ Header with developer info
# ---------------------------
st.title("🧠 Advanced Semantic Similarity Checker v3.0")
st.markdown("""
<div style='text-align: center; padding: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='color: white; margin: 0;'>Developed by <a href='https://www.linkedin.com/in/amal-alexander-305780131/' target='_blank' style='color: #FFD700; text-decoration: none;'>Amal Alexander</a></h4>
    <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>Advanced Multi-Modal Semantic Analysis Engine</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# ✅ Advanced model loading with multiple options
# ---------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        # Primary high-quality model
        models['primary'] = SentenceTransformer("all-mpnet-base-v2")
        # Secondary model for comparison
        models['secondary'] = SentenceTransformer("all-MiniLM-L12-v2")
        # Specialized model for semantic similarity
        models['semantic'] = SentenceTransformer("paraphrase-mpnet-base-v2")
    except Exception as e:
        st.error(f"Model loading error: {e}")
        # Fallback to basic model
        models['primary'] = SentenceTransformer("all-MiniLM-L6-v2")
    return models

models = load_models()

# Initialize NLTK components
@st.cache_resource
def init_nltk():
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return stop_words, lemmatizer
    except:
        return set(), None

stop_words, lemmatizer = init_nltk()

# ---------------------------
# ✅ Advanced text preprocessing
# ---------------------------
def advanced_text_preprocessing(text):
    """Advanced text preprocessing with multiple techniques"""
    if not text or len(text.strip()) < 10:
        return ""
    
    # Remove HTML entities and normalize whitespace
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs, emails, and phone numbers
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Normalize text while preserving important punctuation
    text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
    text = re.sub(r'\d+', 'NUM', text)  # Replace numbers with placeholder
    
    return text.strip()

def extract_semantic_features(text):
    """Extract advanced semantic features from text"""
    if not text:
        return {}
    
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter stopwords and lemmatize
    if lemmatizer:
        meaningful_words = [lemmatizer.lemmatize(word) for word in words 
                          if word.isalpha() and word not in stop_words and len(word) > 2]
    else:
        meaningful_words = [word for word in words 
                          if word.isalpha() and len(word) > 2]
    
    # Calculate various metrics
    features = {
        'sentence_count': len(sentences),
        'word_count': len(words),
        'meaningful_word_count': len(meaningful_words),
        'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
        'vocabulary_richness': len(set(meaningful_words)) / len(meaningful_words) if meaningful_words else 0,
        'word_frequency': Counter(meaningful_words)
    }
    
    return features

# ---------------------------
# ✅ Enhanced content extraction with structure preservation
# ---------------------------
def extract_structured_content(soup):
    """Extract structured content with semantic importance weighting"""
    content_blocks = []
    
    # Title (highest weight)
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        content_blocks.append({
            'type': 'title',
            'content': title,
            'weight': 3.0,
            'text': f"[TITLE]: {title}"
        })
    
    # Meta description (high weight)
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        desc = meta_desc['content'].strip()
        content_blocks.append({
            'type': 'meta_description',
            'content': desc,
            'weight': 2.5,
            'text': f"[META]: {desc}"
        })
    
    # Headers with hierarchical weighting
    header_weights = {'h1': 2.5, 'h2': 2.0, 'h3': 1.8, 'h4': 1.5, 'h5': 1.3, 'h6': 1.1}
    for tag, weight in header_weights.items():
        for header in soup.find_all(tag):
            text = header.get_text(strip=True)
            if text and len(text) > 3:
                content_blocks.append({
                    'type': tag,
                    'content': text,
                    'weight': weight,
                    'text': f"[{tag.upper()}]: {text}"
                })
    
    # Main content areas
    main_content_selectors = [
        'main', 'article', '.content', '.main-content', 
        '.post-content', '.entry-content', '#content'
    ]
    
    main_text = ""
    for selector in main_content_selectors:
        elements = soup.select(selector)
        if elements:
            main_text = ' '.join([el.get_text(strip=True) for el in elements])
            break
    
    # Paragraphs (with smart filtering)
    paragraphs = soup.find_all('p')
    for i, p in enumerate(paragraphs[:20]):  # Limit to first 20 paragraphs
        text = p.get_text(strip=True)
        if text and len(text) > 50:  # Filter short paragraphs
            weight = 1.2 if i < 5 else 1.0  # Higher weight for early paragraphs
            content_blocks.append({
                'type': 'paragraph',
                'content': text,
                'weight': weight,
                'text': f"[P]: {text}"
            })
    
    # Lists (structured data)
    for ul in soup.find_all(['ul', 'ol'])[:10]:
        items = [li.get_text(strip=True) for li in ul.find_all('li')]
        if items:
            list_text = ' '.join(items)
            content_blocks.append({
                'type': 'list',
                'content': list_text,
                'weight': 1.1,
                'text': f"[LIST]: {list_text}"
            })
    
    return content_blocks

def extract_advanced_content(url):
    """Advanced content extraction with error handling and retries"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # Multiple retry attempts
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
                response.raise_for_status()
                break
            except requests.exceptions.Timeout:
                if attempt == 2:
                    return {"error": "Timeout after 3 attempts", "blocks": [], "raw_text": ""}
                time.sleep(2)
            except Exception as e:
                if attempt == 2:
                    return {"error": str(e), "blocks": [], "raw_text": ""}
                time.sleep(1)
        
        # Parse content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Extract structured content
        content_blocks = extract_structured_content(soup)
        
        # Get raw text for fallback
        raw_text = soup.get_text(separator=' ', strip=True)
        raw_text = advanced_text_preprocessing(raw_text)
        
        return {
            "error": None,
            "blocks": content_blocks,
            "raw_text": raw_text,
            "url": url
        }
        
    except Exception as e:
        return {"error": str(e), "blocks": [], "raw_text": "", "url": url}

# ---------------------------
# ✅ Multi-model similarity computation
# ---------------------------
def compute_multi_model_similarity(text1, text2, use_ensemble=True):
    """Compute similarity using multiple models and techniques"""
    if not text1 or not text2 or len(text1.strip()) < 10 or len(text2.strip()) < 10:
        return {"error": "Insufficient content", "scores": {}}
    
    try:
        scores = {}
        
        # Model-based similarities
        for model_name, model in models.items():
            try:
                embeddings = model.encode([text1, text2], convert_to_tensor=True)
                similarity = float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
                scores[f'{model_name}_semantic'] = similarity
            except Exception as e:
                scores[f'{model_name}_semantic'] = 0.0
        
        # TF-IDF based similarity
        try:
            vectorizer = TfidfVectorizer(
                max_features=5000, 
                stop_words='english', 
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95
            )
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores['tfidf_similarity'] = float(tfidf_similarity)
        except:
            scores['tfidf_similarity'] = 0.0
        
        # Jaccard similarity (word level)
        try:
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
            scores['jaccard_similarity'] = jaccard
        except:
            scores['jaccard_similarity'] = 0.0
        
        # Semantic feature similarity
        try:
            features1 = extract_semantic_features(text1)
            features2 = extract_semantic_features(text2)
            
            # Structure similarity
            structure_sim = 1 - abs(features1['sentence_count'] - features2['sentence_count']) / max(features1['sentence_count'], features2['sentence_count'], 1)
            scores['structure_similarity'] = structure_sim
            
            # Vocabulary overlap
            if features1['word_frequency'] and features2['word_frequency']:
                common_words = set(features1['word_frequency'].keys()) & set(features2['word_frequency'].keys())
                vocab_overlap = len(common_words) / len(set(features1['word_frequency'].keys()) | set(features2['word_frequency'].keys()))
                scores['vocabulary_overlap'] = vocab_overlap
            else:
                scores['vocabulary_overlap'] = 0.0
        except:
            scores['structure_similarity'] = 0.0
            scores['vocabulary_overlap'] = 0.0
        
        # Ensemble score (weighted average)
        if use_ensemble and scores:
            weights = {
                'primary_semantic': 0.35,
                'semantic_semantic': 0.25,
                'tfidf_similarity': 0.2,
                'secondary_semantic': 0.1,
                'jaccard_similarity': 0.05,
                'vocabulary_overlap': 0.05
            }
            
            ensemble_score = sum(scores.get(metric, 0) * weight for metric, weight in weights.items())
            scores['ensemble_score'] = ensemble_score
        
        return {"error": None, "scores": scores}
        
    except Exception as e:
        return {"error": str(e), "scores": {}}

def compute_block_similarity(blocks1, blocks2, threshold=0.75):
    """Advanced block-level similarity with weighted importance"""
    if not blocks1 or not blocks2:
        return []
    
    similar_blocks = []
    
    # Create embeddings for all blocks
    try:
        texts1 = [block['text'] for block in blocks1]
        texts2 = [block['text'] for block in blocks2]
        
        embeddings1 = models['primary'].encode(texts1, convert_to_tensor=True)
        embeddings2 = models['primary'].encode(texts2, convert_to_tensor=True)
        
        similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)
        
        for i, block1 in enumerate(blocks1):
            for j, block2 in enumerate(blocks2):
                base_similarity = similarity_matrix[i][j].item()
                
                # Apply weight-based adjustment
                weight_factor = (block1['weight'] + block2['weight']) / 4.0  # Normalize
                adjusted_similarity = base_similarity * (0.8 + 0.2 * weight_factor)  # Boost important content
                
                if adjusted_similarity >= threshold:
                    similar_blocks.append({
                        'block1': block1,
                        'block2': block2,
                        'similarity': round(adjusted_similarity, 4),
                        'base_similarity': round(base_similarity, 4),
                        'importance': round(weight_factor, 2)
                    })
    
    except Exception as e:
        st.error(f"Block similarity computation error: {e}")
    
    # Sort by similarity score
    similar_blocks.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_blocks

# ---------------------------
# ✅ Enhanced Streamlit UI
# ---------------------------
# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Settings")
    use_ensemble = st.checkbox("Use Ensemble Scoring", value=True, help="Combines multiple similarity metrics for higher accuracy")
    
    # Analysis depth
    analysis_depth = st.selectbox(
        "Analysis Depth",
        ["Quick Scan", "Standard Analysis", "Deep Analysis"],
        index=1,
        help="Higher depth = more accurate but slower"
    )
    
    # Thresholds
    st.subheader("Similarity Thresholds")
    page_threshold = st.slider("Page Similarity Threshold", 0.5, 1.0, 0.85, 0.01)
    block_threshold = st.slider("Block Similarity Threshold", 0.5, 1.0, 0.75, 0.01)
    
    # Advanced options
    st.subheader("Advanced Options")
    max_urls = st.number_input("Max URLs to Process", min_value=2, max_value=5000, value=1000)
    timeout_seconds = st.number_input("Request Timeout (seconds)", min_value=5, max_value=30, value=15)

# Main interface
with st.expander("📚 How to Use v3.0", expanded=False):
    st.markdown("""
    ### 🚀 Advanced Features in v3.0:
    - **Multi-Model Ensemble**: Uses multiple SOTA transformer models
    - **Semantic Feature Analysis**: Deep structural and vocabulary analysis
    - **Weighted Content Importance**: Titles and headers get higher priority
    - **Smart Preprocessing**: Advanced text cleaning and normalization
    - **Error Recovery**: Robust handling with multiple retry attempts
    - **Detailed Analytics**: Comprehensive similarity breakdowns
    
    ### 📋 Instructions:
    1. **Input URLs**: Paste URLs or upload a file (supports .txt, .csv, .xlsx)
    2. **Configure Settings**: Use sidebar to adjust analysis parameters
    3. **Select Analysis Mode**: Choose between speed and accuracy
    4. **Review Results**: Get detailed similarity scores and explanations
    5. **Export Data**: Download comprehensive reports
    """)

# URL Input Section
st.header("📥 URL Input")
col1, col2 = st.columns([3, 1])

with col1:
    input_method = st.radio("Input Method", ["Paste URLs", "Upload File"], horizontal=True)

urls = []

if input_method == "Paste URLs":
    url_text = st.text_area(
        "Paste URLs (one per line)", 
        height=200, 
        placeholder="https://example1.com\nhttps://example2.com\n..."
    )
    if url_text:
        urls = [line.strip() for line in url_text.splitlines() if line.strip()]

else:
    uploaded_file = st.file_uploader(
        "Upload File", 
        type=["txt", "csv", "xlsx"], 
        help="Upload a file containing URLs"
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".txt"):
                content = uploaded_file.read().decode("utf-8")
                urls = [line.strip() for line in content.splitlines() if line.strip()]
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                urls = df.iloc[:, 0].dropna().astype(str).tolist()
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                urls = df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Process URLs
if urls:
    urls = list(dict.fromkeys(urls))  # Remove duplicates
    if len(urls) > max_urls:
        st.warning(f"⚠️ Limiting to {max_urls} URLs")
        urls = urls[:max_urls]
    
    st.success(f"✅ {len(urls)} URLs loaded")
    with st.expander("Preview URLs", expanded=False):
        st.dataframe(pd.DataFrame(urls, columns=["URLs"]))

# Analysis Mode Selection
st.header("🔬 Analysis Configuration")
col1, col2 = st.columns(2)

with col1:
    comparison_mode = st.selectbox(
        "Comparison Mode",
        ["All vs All", "Sequential Pairs", "One vs Rest"],
        help="All vs All: Complete comparison matrix (recommended for ≤10 URLs)\nSequential: Adjacent pairs only\nOne vs Rest: First URL against all others"
    )

with col2:
    detail_level = st.selectbox(
        "Detail Level",
        ["Page-Level Only", "Block-Level Analysis", "Full Deep Analysis"],
        index=1
    )

# Run Analysis
if st.button("🚀 Start Advanced Analysis", use_container_width=True, type="primary"):
    if len(urls) < 2:
        st.error("Please provide at least 2 URLs")
    else:
        # Estimate processing time
        if comparison_mode == "All vs All":
            total_pairs = len(urls) * (len(urls) - 1) // 2
        elif comparison_mode == "Sequential Pairs":
            total_pairs = len(urls) - 1
        else:  # One vs Rest
            total_pairs = len(urls) - 1
        
        time_per_pair = 3 if detail_level == "Page-Level Only" else 8
        estimated_time = total_pairs * time_per_pair
        
        st.info(f"Processing {total_pairs} pairs. Estimated time: {estimated_time//60}m {estimated_time%60}s")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        results = []
        processed = 0
        
        # Generate URL pairs based on mode
        if comparison_mode == "Sequential Pairs":
            # For complete coverage, use All vs All for small sets, Sequential for large sets
            if len(urls) <= 10:
                url_pairs = [(urls[i], urls[j]) for i in range(len(urls)) for j in range(i+1, len(urls))]
                st.info(f"📊 Using All vs All mode for complete coverage ({len(urls)} URLs)")
            else:
                url_pairs = [(urls[i], urls[i+1]) for i in range(len(urls)-1)]
                st.warning(f"⚠️ Sequential mode: Some pairs will be missed. Consider 'All vs All' for complete analysis.")
        elif comparison_mode == "All vs All":
            url_pairs = [(urls[i], urls[j]) for i in range(len(urls)) for j in range(i+1, len(urls))]
        else:  # One vs Rest
            url_pairs = [(urls[0], urls[i]) for i in range(1, len(urls))]
        
        # Process each pair
        for url1, url2 in url_pairs:
            status_text.text(f"Processing: {urlparse(url1).netloc} vs {urlparse(url2).netloc}")
            
            # Extract content
            content1 = extract_advanced_content(url1)
            content2 = extract_advanced_content(url2)
            
            if content1["error"] or content2["error"]:
                result = {
                    "URL 1": url1,
                    "URL 2": url2,
                    "Status": "❌ Error",
                    "Error": content1["error"] or content2["error"],
                    "Overall Score": 0,
                    "Scores": {}
                }
            else:
                # Compute similarities
                similarity_result = compute_multi_model_similarity(
                    content1["raw_text"], 
                    content2["raw_text"], 
                    use_ensemble
                )
                
                if similarity_result["error"]:
                    overall_score = 0
                    scores = {}
                    status = "❌ Analysis Error"
                else:
                    scores = similarity_result["scores"]
                    overall_score = scores.get('ensemble_score', scores.get('primary_semantic', 0))
                    status = "🔥 High Similarity" if overall_score >= page_threshold else "✅ Processed"
                
                result = {
                    "URL 1": url1,
                    "URL 2": url2,
                    "Status": status,
                    "Overall Score": round(overall_score, 4),
                    "Primary Model": round(scores.get('primary_semantic', 0), 4),
                    "Semantic Model": round(scores.get('semantic_semantic', 0), 4),
                    "TF-IDF Score": round(scores.get('tfidf_similarity', 0), 4),
                    "Ensemble Score": round(scores.get('ensemble_score', 0), 4),
                    "Content 1": content1["raw_text"][:500] if content1["raw_text"] else "",
                    "Content 2": content2["raw_text"][:500] if content2["raw_text"] else ""
                }
                
                # Block-level analysis if requested
                if detail_level != "Page-Level Only" and not (content1["error"] or content2["error"]):
                    similar_blocks = compute_block_similarity(
                        content1["blocks"], 
                        content2["blocks"], 
                        block_threshold
                    )
                    result["Similar Blocks"] = similar_blocks
                    result["Block Matches"] = len(similar_blocks)
            
            results.append(result)
            processed += 1
            progress_bar.progress(processed / total_pairs)
        
        # Display results
        status_text.text("✅ Analysis Complete!")
        progress_bar.progress(1.0)
        
        # Create results DataFrame
        display_columns = ["URL 1", "URL 2", "Status", "Overall Score", "Primary Model", "TF-IDF Score"]
        if use_ensemble:
            display_columns.append("Ensemble Score")
        if detail_level != "Page-Level Only":
            display_columns.append("Block Matches")
        
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pairs", len(results))
        with col2:
            high_sim = len([r for r in results if r.get("Overall Score", 0) >= page_threshold])
            st.metric("High Similarity", high_sim)
        with col3:
            avg_score = np.mean([r.get("Overall Score", 0) for r in results])
            st.metric("Average Score", f"{avg_score:.3f}")
        with col4:
            errors = len([r for r in results if "Error" in r.get("Status", "")])
            st.metric("Errors", errors)
        
        # Results table
        st.subheader("📊 Results Overview")
        st.dataframe(
            results_df[display_columns], 
            use_container_width=True,
            column_config={
                "Overall Score": st.column_config.ProgressColumn(
                    "Overall Score",
                    help="Combined similarity score",
                    min_value=0,
                    max_value=1,
                ),
                "Status": st.column_config.TextColumn("Status", width="medium")
            }
        )
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Report",
                csv_data,
                f"similarity_analysis_v3_{int(time.time())}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Detailed JSON export
            json_data = pd.DataFrame(results).to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download Detailed JSON",
                json_data,
                f"detailed_analysis_v3_{int(time.time())}.json",
                "application/json",
                use_container_width=True
            )
        
        # Detailed analysis view
        if detail_level != "Page-Level Only":
            with st.expander("🔍 Detailed Block Analysis"):
                for i, result in enumerate(results):
                    if result.get("Similar Blocks"):
                        st.markdown(f"### Pair {i+1}: {result['Block Matches']} Similar Blocks")
                        st.markdown(f"**URLs**: {result['URL 1'][:60]}... vs {result['URL 2'][:60]}...")
                        
                        for j, block_match in enumerate(result["Similar Blocks"][:5]):  # Show top 5
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Block {j+1}A** (Score: {block_match['similarity']})")
                                st.markdown(f"*Type: {block_match['block1']['type']}*")
                                st.text_area("", block_match['block1']['content'][:300], height=100, key=f"detail_{i}_{j}_1")
                            with col2:
                                st.markdown(f"**Block {j+1}B** (Importance: {block_match['importance']})")
                                st.markdown(f"*Type: {block_match['block2']['type']}*")
                                st.text_area("", block_match['block2']['content'][:300], height=100, key=f"detail_{i}_{j}_2")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #666; margin: 0;'>Advanced Semantic Similarity Checker v3.0</p>
    <p style='color: #666; margin: 5px 0 0 0;'>Powered by State-of-the-Art Transformer Models | Developed by <a href='https://www.linkedin.com/in/amal-alexander-305780131/' target='_blank'>Amal Alexander</a></p>
</div>
""", unsafe_allow_html=True)