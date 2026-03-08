from inference import EmailCategorizer

categorizer = EmailCategorizer()
result = categorizer.predict('urgent server down')
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
