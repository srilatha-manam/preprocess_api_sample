from sentence_transformers import SentenceTransformer

# Load pre-trained SentenceTransformer model
def get_text_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
