
#code to test dialogue similarity on raw input
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CSV file path
CSV_PATH = "C:\\Users\\Srilatha\\Desktop\\preprocess_api_sample\\preprocess_api_sample\\data\\dialouge\\dialouges.csv"

class DialogueProcessor:
    def __init__(self):
        """Initialize and load dialogues."""
        self.dialogues = self._load_dialogues()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.dialogues) if self.dialogues else None

    def _load_dialogues(self):
        """Load dialogues from CSV file."""
        try:
            df = pd.read_csv(CSV_PATH)
            if "dialogue" not in df.columns:
                raise ValueError("CSV file must contain a 'dialogue' column")
            return df["dialogue"].dropna().tolist()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []

    def find_most_similar(self, prompt: str) -> str:
        """Find the most similar dialogue to the given prompt."""
        if not self.dialogues:
            return "No dialogues available."

        prompt_tfidf = self.vectorizer.transform([prompt])
        similarities = cosine_similarity(prompt_tfidf, self.tfidf_matrix)
        best_match_index = np.argmax(similarities)

        return self.dialogues[best_match_index]


# Create an instance of DialogueProcessor to be used in the API
dialogue_processor = DialogueProcessor()
