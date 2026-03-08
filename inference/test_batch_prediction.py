# inference/test_batch_prediction.py

from inference.batch_processor import BatchProcessor
import os

def test_batch_prediction():
    """Test batch email prediction"""
   
    # Sample emails for batch processing
    test_emails = [
        {
            'subject': 'Quarterly Sales Report Q4 2023',
            'body': 'Please find attached the sales performance metrics for Q4. Revenue increased by 15%.'
        },
        {
            'subject': 'Your Amazon Order #12345 Has Shipped',
            'body': 'Your order is on the way. Track your package with tracking number XYZ789.'
        },
        {
            'subject': 'Team Meeting Tomorrow at 2 PM',
            'body': 'Reminder: Project review meeting in Conference Room B. Please bring status updates.'
        },
        {
            'subject': 'URGENT: Account Security Alert',
            'body': 'We detected unusual activity. Click here to verify your account immediately.'
        },
        {
            'subject': 'Invoice #INV-2024-001 - Payment Due',
            'body': 'Your invoice for January services is attached. Payment due by Feb 15th.'
        }
    ]
   
    # Initialize batch processor (uses active model from registry)
    processor = BatchProcessor()

    print("=" * 60)
    print("BATCH PREDICTION TEST")
    print("=" * 60)
    print(f"\nProcessing {len(test_emails)} emails...\n")

    # Convert to expected format (id + text)
    emails_for_processing = [
        {'id': i + 1, 'text': item['body'], 'subject': item['subject']}
        for i, item in enumerate(test_emails)
    ]

    # Process batch (do not persist to DB for this demo)
    results = processor.process_emails(emails_for_processing, save_to_db=False)

    # Display results
    for i, (email, result) in enumerate(zip(emails_for_processing, results), 1):
        print(f"\n--- Email {i} ---")
        print(f"Subject: {email['subject']}" )
        print(f"Category: {result.get('category')}")
        print(f"Confidence: {result.get('confidence'):.2%}")
        if 'save_error' in result:
            print(f"Save error: {result['save_error']}")
        if result.get('low_confidence'):
            print("Low confidence warning")
   
    # Summary statistics
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results if 'save_error' not in r)
    failed = sum(1 for r in results if 'save_error' in r)

    print(f"Total emails: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Category distribution
    from collections import Counter
    categories = [r['category'] for r in results if 'save_error' not in r]
    category_counts = Counter(categories)

    print(f"\nCategory Distribution:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")

if __name__ == "__main__":
    test_batch_prediction()