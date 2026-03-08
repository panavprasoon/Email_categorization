# generate synthetic dataset
import pandas as pd
import random

# Email templates per category
templates = {
    'incident': [
        "URGENT: {} service down. Immediate action required.",
        "CRITICAL: {} failure detected. Investigation ongoing.",
        "Production issue: {} not responding. Emergency response needed.",
        "System alert: {} experiencing errors. DevOps investigating.",
        "Security incident: {} vulnerability found. Patch needed urgently."
    ],
    'meeting': [
        "Team meeting scheduled for {} to discuss {}.",
        "Planning session for {} project next week.",
        "{} review meeting moved to Thursday.",
        "Kickoff meeting for {} initiative on Monday.",
        "Weekly sync for {} team at 10am tomorrow."
    ],
    'report': [
        "{} quarterly report attached for review.",
        "Monthly {} analysis completed. Summary attached.",
        "{} performance metrics for last week.",
        "Annual {} report ready for distribution.",
        "Weekly {} status update and progress report."
    ],
    'spam': [
        "Congratulations! You won {}! Click here now!",
        "Amazing {} opportunity! Limited time offer!",
        "Get {} free! Act now before it's too late!",
        "You've been selected for {}! Claim your prize!",
        "Unbelievable {} deal! 90% discount today only!"
    ],
    'approval': [
        "Approval needed for {} budget increase.",
        "Please approve {} request by end of week.",
        "{} purchase order requires manager approval.",
        "Sign-off needed for {} contract renewal.",
        "Budget approval: {} project funding request."
    ],
    'reminder': [
        "Reminder: {} deadline is next Friday.",
        "Don't forget: {} submission due tomorrow.",
        "Friendly reminder about {} meeting at 2pm.",
        "Reminder: {} review needs to be completed.",
        "Please remember to submit {} by EOD."
    ]
}

# Variables to fill templates
services = ['database', 'API', 'authentication', 'payment', 'email', 'cloud']
topics = ['budget', 'architecture', 'hiring', 'project X', 'migration', 'security']
items = ['$10,000', 'iPhone', 'vacation', 'prize', 'credit card', 'investment']
tasks = ['timesheets', 'expenses', 'OKRs', 'reviews', 'reports', 'surveys']

def generate_emails(count=200):
    """Generate synthetic email dataset."""
    emails = []
    email_id = 1
    
    emails_per_category = count // len(templates)
    
    for category, template_list in templates.items():
        for i in range(emails_per_category):
            template = random.choice(template_list)
            
            # Fill template with random variables
            if category == 'incident':
                text = template.format(random.choice(services))
            elif category == 'meeting':
                text = template.format(random.choice(topics), random.choice(topics))
            elif category == 'report':
                text = template.format(random.choice(topics))
            elif category == 'spam':
                text = template.format(random.choice(items))
            elif category == 'approval':
                text = template.format(random.choice(topics))
            elif category == 'reminder':
                text = template.format(random.choice(tasks))
            
            emails.append({
                'email_id': email_id,
                'text': text,
                'category': category
            })
            email_id += 1
    
    # Shuffle emails
    random.shuffle(emails)
    
    # Create DataFrame
    df = pd.DataFrame(emails)
    return df

if __name__ == "__main__":
    # Generate 200 emails
    df = generate_emails(200)
    
    # Save to CSV
    df.to_csv('data/sample_emails.csv', index=False)
    
    # Print statistics
    print(f"Generated {len(df)} emails")
    print("\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    print(f"\nSaved to: data/sample_emails.csv")
