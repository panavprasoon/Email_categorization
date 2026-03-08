# text cleaning

"""
Text preprocessing utilities for email categorization.

This module provides the TextPreprocessor class which handles:
- Lowercase conversion
- URL and email address removal
- Punctuation and number removal
- Stopword filtering
- Lemmatization
- Custom word length constraints

CRITICAL: Preprocessing must be IDENTICAL for training and inference.
Always use the same configuration for both phases.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """
    Configurable text preprocessing for email content.
    
    Features:
    - Lowercase conversion
    - URL/email removal
    - Punctuation/number removal
    - Stopword filtering
    - Lemmatization
    - Word length filtering
    
    Example:
        >>> preprocessor = TextPreprocessor(
        ...     lowercase=True,
        ...     remove_stopwords=True,
        ...     apply_lemmatization=True
        ... )
        >>> text = "Check this URL: https://example.com for details!"
        >>> preprocessor.clean_text(text)
        'check url detail'
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_stopwords: bool = True,
        apply_lemmatization: bool = True,
        min_word_length: int = 2,
        max_word_length: Optional[int] = None,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize text preprocessor with configuration.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation characters
            remove_numbers: Remove numeric digits
            remove_urls: Remove HTTP/HTTPS URLs
            remove_emails: Remove email addresses
            remove_stopwords: Remove common stopwords (English)
            apply_lemmatization: Apply word lemmatization
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep (None = no limit)
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_stopwords = remove_stopwords
        self.apply_lemmatization = apply_lemmatization
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Initialize stopwords
        self.stopwords_set = set()
        if self.remove_stopwords:
            try:
                self.stopwords_set = set(stopwords.words('english'))
            except LookupError:
                print("Warning: NLTK stopwords not found. Run download_nltk_data.py")
                self.stopwords_set = set()
        
        # Add custom stopwords
        if custom_stopwords:
            self.stopwords_set.update(custom_stopwords)
        
        # Initialize lemmatizer
        self.lemmatizer = None
        if self.apply_lemmatization:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                print("Warning: NLTK wordnet not found. Run download_nltk_data.py")
                self.lemmatizer = None
        
        # Regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_text("Visit http://example.com NOW!")
            'visit'
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback if punkt not available
            tokens = text.split()
        
        # Filter stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords_set]
        
        # Apply lemmatization
        if self.apply_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Filter by word length
        tokens = [
            t for t in tokens
            if len(t) >= self.min_word_length
            and (self.max_word_length is None or len(t) <= self.max_word_length)
        ]
        
        # Join tokens back into string
        return ' '.join(tokens)
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of text strings.
        
        Args:
            texts: List of raw texts to preprocess
            
        Returns:
            List of cleaned texts
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> texts = ["Hello world!", "Test email@example.com"]
            >>> preprocessor.clean_batch(texts)
            ['hello world', 'test']
        """
        return [self.clean_text(text) for text in texts]
    
    def get_config(self) -> dict:
        """
        Get preprocessor configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'lowercase': self.lowercase,
            'remove_punctuation': self.remove_punctuation,
            'remove_numbers': self.remove_numbers,
            'remove_urls': self.remove_urls,
            'remove_emails': self.remove_emails,
            'remove_stopwords': self.remove_stopwords,
            'apply_lemmatization': self.apply_lemmatization,
            'min_word_length': self.min_word_length,
            'max_word_length': self.max_word_length
        }
    
    def __repr__(self) -> str:
        """String representation of preprocessor configuration."""
        config = self.get_config()
        config_str = ', '.join(f"{k}={v}" for k, v in config.items())
        return f"TextPreprocessor({config_str})"
