import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

def ai_paragraph_wrangle_engine(raw_text, words_per_paragraph=40):
    """
    Simulates advanced NLP semantic chunking. 
    Breaks unstructured text data down into structured, evenly distributed paragraphs.
    """
    # Clean up dirty layout breaks, structural tabs, and double spaces
    sanitized_text = " ".join(raw_text.split())
    words = sanitized_text.split()
    
    if not words:
        return []
        
    # --- DATA WRANGLING VIA PANDAS ---
    # Convert word array into a structured Pandas series to manipulate positions natively
    word_series = pd.Series(words)
    
    # Calculate group indexes mathematically to determine paragraph break splits
    chunk_indices = word_series.index // words_per_paragraph
    
    # Group the words using Pandas matrix grouping and rejoin into clear paragraphs
    paragraphs = word_series.groupby(chunk_indices).apply(lambda x: " ".join(x)).tolist()
    
    formatted_paragraphs = []
    for idx, paragraph in enumerate(paragraphs):
        # Apply standard grammatical formatting rules natively
        paragraph = paragraph.strip().capitalize()
        if not paragraph.endswith(('.', '!', '?')):
            paragraph += '.'
            
        formatted_paragraphs.append({
            "paragraph_number": idx + 1,
            "word_count": len(paragraph.split()),
            "text": paragraph
        })
        
    return formatted_paragraphs


@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "active", "engine": "Python/Flask AI Text Processing Architecture"})


@app.route('/format-text', methods=['POST'])
def format_text():
    try:
        # force=True acts as an automated bypass for broken Content-Type headers
        data = request.get_json(force=True)
        
        if not data or 'raw_text' not in data:
            return jsonify({"error": "Processing aborted: Field 'raw_text' is missing or blank."}), 400
            
        unstructured_payload = str(data.get('raw_text', '')).strip()
        target_density = int(data.get('paragraph_density', 40))
        
        if not unstructured_payload:
            return jsonify({"error": "Cannot format an empty text string."}), 400

        # Run text structural parsing
        processed_chunks = ai_paragraph_wrangle_engine(unstructured_payload, words_per_paragraph=target_density)
        
        return jsonify({
            "message": "Unstructured text payload successfully restructured into semantic paragraph layouts.",
            "total_paragraphs_generated": len(processed_chunks),
            "formatted_document": processed_chunks
        })
        
    except Exception as e:
        return jsonify({"error": f"Server processing breakdown: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
