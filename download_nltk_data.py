# language data
import nltk
import sys

def download_nltk_data():
    """Download all required NLTK data packages."""
    packages = [
        'punkt',           # Tokenizer
        'stopwords',       # Stopwords list
        'wordnet',         # WordNet lexical database
        'averaged_perceptron_tagger'  # POS tagger
    ]
    
    print("Downloading NLTK data packages...")
    print("=" * 60)
    
    for package in packages:
        try:
            print(f"\nDownloading '{package}'...")
            nltk.download(package, quiet=False)
            print(f"✓ Successfully downloaded '{package}'")
        except Exception as e:
            print(f"✗ Error downloading '{package}': {e}", file=sys.stderr)
            return False
    
    print("\n" + "=" * 60)
    print("All NLTK data packages downloaded successfully!")
    print("You can now use the preprocessing pipeline.")
    return True

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)

