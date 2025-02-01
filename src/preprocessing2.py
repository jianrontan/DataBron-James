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
        try:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger')

            self.stop_words = set(stopwords.words('english'))
            self.nlp = spacy.load("en_core_web_sm")
            self.summarizer = LsaSummarizer()
        except Exception as e:
            print(f"Error initializing NLTK components: {str(e)}")
            raise

    def _summarize_text(self, text, sentences_count=1):
        """Generate summary using LSA with fallback"""
        try:
            if not text or len(text.strip()) == 0:
                return text

            parser = PlaintextParser.from_string(
                text,
                Tokenizer("english")
            )

            self.summarizer.stop_words = self.stop_words

            summary = self.summarizer(parser.document, sentences_count)
            return " ".join([str(sentence) for sentence in summary])
        except Exception as e:
            print(f"Error in summarization, returning original text: {str(e)}")
            sentences = self.nlp(text).sents
            return " ".join([str(sent) for sent in list(sentences)[:sentences_count]])

    def load_and_clean_text(self, file_path):
        """Load and clean text from Excel file"""
        try:
            df = pd.read_excel(file_path)

            # Handle missing values
            if df['Text'].isnull().any():
                df['Text'].fillna('', inplace=True)

            # Create cleaned words column
            df['Cleaned_words'] = df['Text']

            # Apply cleaning operations
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

    def apply_ner(self, tokens):
        """Apply Named Entity Recognition"""
        sentence = " ".join(tokens)
        doc = self.nlp(sentence)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def process_text(self, file_path):
        """Process text with added summarization"""
        try:
            data = self.load_and_clean_text(file_path)
            if data is None:
                return None

            print("Creating spaCy documents...")
            data['doc'] = list(self.nlp.pipe(
                data['Cleaned_words'], batch_size=32))

            data['Tokenized'] = data['doc'].apply(
                lambda x: [token.text for token in x]
            )

            data['NER_Entities'] = data['doc'].apply(
                lambda x: [(ent.text, ent.label_) for ent in x.ents]
            )

            print("Generating summaries...")
            data['Summary'] = data['Text'].apply(self._summarize_text)

            data['Summary_Tokenized'] = data['Summary'].apply(
                lambda x: [token.text for token in self.nlp(x)]
            )

            data['Summary_Entities'] = data['Summary'].apply(
                lambda x: [(ent.text, ent.label_) for ent in self.nlp(x).ents]
            )

            if 'Link' not in data.columns:
                data['Link'] = ''

            return data

        except Exception as e:
            print(f"Error in processing pipeline: {str(e)}")
            return None
