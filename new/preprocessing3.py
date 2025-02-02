import nltk
import os
import numpy as np
import pandas as pd
import re
import spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        """Initialize the TextPreprocessor with required NLTK and spaCy components"""
        # Download required NLTK data
        nltk.download('stopwords')
        nltk.download('punkt')

        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")

    def save_to_csv(self, df, output_path):
        """Save processed data to CSV"""
        try:
            if df is not None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False, encoding="utf-8")
                print(f"✅ Processed data saved to: {output_path}")
            else:
                print("❌ No data to save.")
        except Exception as e:
            print(f"❌ Error saving to CSV: {str(e)}")  

    def load_and_clean_text(self, file_path):
        """Load and clean text from TXT or Excel file"""
        try:
            if file_path.endswith(".txt"):
                df = self._load_txt_file(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
                # Ensure 'Sentence' column exists
                if 'Sentence' not in df.columns:
                    raise ValueError("❌ Missing required 'Sentence' column in Excel file.")
                df['Sentence'].fillna('', inplace=True)
            else:
                raise ValueError("❌ Unsupported file format. Please provide a .txt or .xlsx file.")

            # Apply text cleaning
            df["Sentence"] = df["Sentence"].apply(self._clean_text)
            return df

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {str(e)}")
            return None

    def _load_txt_file(self, file_path):
        """Load and process a TXT file (formatted as tab-separated)"""
        try:
            df = pd.read_csv(file_path, delimiter="\t", header=None, names=["Entity1", "Predicate", "Entity2"])
            df["Sentence"] = df["Entity1"] + " " + df["Predicate"] + " " + df["Entity2"]
            return df
        except Exception as e:
            print(f"❌ Error loading TXT file {file_path}: {str(e)}")
            return None

    def _clean_text(self, text):
        """Clean individual text"""
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)  # Remove links
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Remove hyphens
        text = re.sub(r'\d+', ' ', text)  # Remove digits
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in self.stop_words])  # Remove stopwords
        return text

    def tokenize_text(self, data):
        """Tokenize cleaned text"""
        try:
            if 'Sentence' not in data.columns:
                raise ValueError("❌ Missing 'Sentence' column in dataset.")
            data['Tokenized'] = data['Sentence'].apply(lambda x: x.split())
            return data
        except Exception as e:
            print(f"❌ Error in tokenization: {str(e)}")
            return None

    def apply_ner(self, tokens):
        """Apply Named Entity Recognition (NER)"""
        if not tokens:
            return []  # Return empty list if no tokens available

        sentence = " ".join(tokens)
        doc = self.nlp(sentence)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def process_text(self, file_path):
        """Complete text processing pipeline"""
        try:
            # Load and clean text
            data = self.load_and_clean_text(file_path)
            if data is None:
                return None

            # Tokenize
            data = self.tokenize_text(data)

            # Apply NER
            data["NER_Entities"] = data["Tokenized"].apply(self.apply_ner)

            return data

        except Exception as e:
            print(f"❌ Error in processing pipeline: {str(e)}")
            return None
