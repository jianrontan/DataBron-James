import nltk
import numpy as np
import pandas as pd
import re
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
# from spacy.pipeline.ner import DEFAULT_NER_MODEL


class TextPreprocessor:
    def __init__(self):
        """Initialize the TextPreprocessor with required NLTK and spaCy components"""
        # Download required NLTK data
        nltk.download('stopwords')
        nltk.download('punkt')

        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")

        # NER configuration
        # self.ner_config = {
        #     "moves": None,
        #     "update_with_oracle_cut_size": 100,
        #     "model": DEFAULT_NER_MODEL,
        #     "incorrect_spans_key": "incorrect_spans",
        # }

    def _summarize_text(self, text):
        """Generate summary for individual text"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=1)
            return " ".join([str(sentence) for sentence in summary])
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return text

    def load_and_clean_text(self, file_path):
        """Load and clean text from Excel file"""
        try:
            df = pd.read_excel(file_path)

            if df['Text'].isnull().any():
                df['Text'].fillna('', inplace=True)

            df['Text'] = df['Text'].apply(lambda x: self._summarize_text(x))

            df['Cleaned_words'] = df['Text']

            df['Cleaned_words'] = df['Cleaned_words'].apply(self._clean_text)

            return df

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None

    def _clean_text(self, text):
        """Clean individual text"""
        # Remove links
        text = re.sub(r'http\S+|www\S+|https\S+',
                      ' ', text, flags=re.MULTILINE)

        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove hyphens
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)

        # Remove digits
        text = re.sub(r'\d+', ' ', text)

        # Convert to lowercase
        text = text.lower()

        # Remove stopwords
        text = ' '.join([word for word in text.split()
                        if word not in self.stop_words])

        return text

    def tokenize_text(self, data):
        """Tokenize cleaned text"""
        try:
            data['Tokenized'] = data['Cleaned_words'].apply(
                lambda x: x.split())
            return data
        except Exception as e:
            print(f"Error in tokenization: {str(e)}")
            return None

    def apply_ner(self, text):
        """Apply Named Entity Recognition"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def process_text(self, file_path):
        try:
            data = self.load_and_clean_text(file_path)
            if data is None:
                return None

            print("Creating spaCy documents...")
            data['doc'] = list(self.nlp.pipe(
                data['Cleaned_words'], batch_size=32))
            
            data['doc2'] = list(self.nlp.pipe(
                data['Text'], batch_size=32))

            data['Tokenized'] = data['doc'].apply(
                lambda x: [token.text for token in x])

            data['NER_Entities'] = data['doc2'].apply(self.apply_ner)

            return data
        except Exception as e:
            print(f"Error in processing pipeline: {str(e)}")
            return None
