# CLI tool for quick email categorization testing.

"""
Command-line tool for categorizing emails.

Usage:
    python scripts/categorize_email.py "urgent server down need help"
    python scripts/categorize_email.py --file emails.txt
    python scripts/categorize_email.py --interactive
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import EmailCategorizer


def categorize_single(email_text: str, detailed: bool = False):
    """Categorize a single email."""
    categorizer = EmailCategorizer()
    
    if detailed:
        result = categorizer.predict_with_details(email_text)
    else:
        result = categorizer.predict(email_text)
    
    print("\n" + "=" * 60)
    print("EMAIL CATEGORIZATION RESULT")
    print("=" * 60)
    print(f"Input: {email_text[:100]}{'...' if len(email_text) > 100 else ''}")
    print(f"\nCategory: {result['category']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Version: {result['model_version']}")
    
    if result.get('low_confidence'):
        print("⚠ WARNING: Low confidence prediction")
    
    if detailed and 'top_3_categories' in result:
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(result['top_3_categories'], 1):
            print(f"  {i}. {pred['category']}: {pred['confidence']:.2%}")
    
    if detailed and 'prediction_time_ms' in result:
        print(f"\nPrediction Time: {result['prediction_time_ms']:.2f}ms")
    
    print("=" * 60 + "\n")


def categorize_from_file(filepath: str):
    """Categorize emails from a file (one per line)."""
    categorizer = EmailCategorizer()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        emails = [line.strip() for line in f if line.strip()]
    
    print(f"\nCategorizing {len(emails)} emails from {filepath}...\n")
    
    results = categorizer.predict_batch(emails)
    
    for i, (email, result) in enumerate(zip(emails, results), 1):
        print(f"{i}. {email[:50]}")
        print(f"   → {result['category']} ({result['confidence']:.2%})")
        if result.get('low_confidence'):
            print(f"   ⚠ Low confidence")
        print()
    
    # Summary
    stats = categorizer.get_statistics()
    print("\nStatistics:")
    print(f"  Total: {len(results)}")
    print(f"  Average time: {stats['average_inference_time_ms']:.2f}ms")


def interactive_mode():
    """Interactive categorization mode."""
    categorizer = EmailCategorizer()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE EMAIL CATEGORIZATION")
    print("=" * 60)
    print("Enter email text (or 'quit' to exit)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            email_text = input("Email: ").strip()
            
            if email_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not email_text:
                continue
            
            result = categorizer.predict(email_text)
            print(f"  → Category: {result['category']}")
            print(f"  → Confidence: {result['confidence']:.2%}")
            
            if result.get('low_confidence'):
                print(f"  → ⚠ Low confidence")
            print()
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Final statistics
    stats = categorizer.get_statistics()
    print(f"\nProcessed {stats['predictions_made']} emails")
    print(f"Average time: {stats['average_inference_time_ms']:.2f}ms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Categorize emails')
    parser.add_argument('email_text', nargs='?', help='Email text to categorize')
    parser.add_argument('--file', '-f', help='File with emails (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed results')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.file:
        categorize_from_file(args.file)
    elif args.email_text:
        categorize_single(args.email_text, detailed=args.detailed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
