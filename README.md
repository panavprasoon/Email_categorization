This is a comprehensive file to common issue that you might face and their troubleshooting guide

8.1 CONNECTION ERRORS
----------------------
Problem: "could not connect to server" or timeout

Solutions:
1. Check internet connection
2. Verify Neon project is not suspended:
   - Go to Neon Dashboard
   - Check project status
   - If suspended, it auto-wakes on connection (takes ~1-2 seconds)

3. Verify connection details in .env:
   - Copy exact values from Neon dashboard
   - Include full hostname (e.g., ep-xxx-xxx.neon.tech)
   - Don't include https:// or postgresql:// in DB_HOST

4. Test connection in Neon SQL Editor first

8.2 AUTHENTICATION ERRORS
--------------------------
Problem: "password authentication failed"

Solutions:
1. Copy password exactly from Neon (it's long and complex)
2. Don't use quotes around password in .env file
3. If password has special characters, ensure .env syntax is correct
4. Reset password in Neon dashboard if needed

8.3 SSL ERRORS
--------------
Problem: "SSL connection required"

Solutions:
1. Ensure DB_SSLMODE=require in .env
2. Check config.py includes sslmode in connection string
3. Don't use sslmode=disable with Neon (it won't work)

8.4 TABLE NOT FOUND ERRORS
---------------------------
Problem: "relation 'emails' does not exist"

Solutions:
1. Verify schema was created:
   - Go to Neon SQL Editor
   - Run: \dt
   - Should show all 7 tables

2. If tables don't exist, run create_schema.sql again in SQL Editor

3. Check you're connecting to correct database (neondb vs email_categorization)

8.5 TIMEOUT ERRORS
------------------
Problem: Connection times out

Solutions:
1. Increase timeout in config.py connect_args:
   'connect_timeout': 30

2. Check firewall/antivirus not blocking connection

3. Try different network (e.g., disable VPN)

8.6 IMPORT ERRORS
-----------------
Problem: "ModuleNotFoundError"

Solutions:
1. Activate virtual environment:
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Verify Python version:
   python --version
   (Should be 3.8+)

8.7 PROJECT SUSPENDED
----------------------
Problem: "Project is suspended"

Neon free tier auto-suspends after inactivity.

Solutions:
1. Wait 1-2 seconds - it auto-wakes
2. Retry connection
3. Or manually wake in dashboard


11.1 IMPORT ERRORS
------------------
Problem: ModuleNotFoundError: No module named 'feature_pipeline'
Solution: Ensure you're running from project root directory where feature_pipeline/ exists

Problem: ImportError: cannot import name 'TextPreprocessor'
Solution: Check feature_pipeline/__init__.py has correct imports

11.2 NLTK DATA ERRORS
---------------------
Problem: LookupError: Resource 'tokenizers/punkt' not found
Solution: Run download_nltk_data.py script:
  python download_nltk_data.py

Problem: Warning about stopwords not found
Solution: NLTK data incomplete. Re-run download script with --force:
  python -c "import nltk; nltk.download('stopwords', quiet=False)"

11.3 SKLEARN ERRORS
-------------------
Problem: AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names_out'
Solution: Update scikit-learn to 1.0+:
  pip install --upgrade scikit-learn==1.3.2

Problem: ValueError: empty vocabulary
Solution: Not enough unique words in training data. Reduce min_df or increase data:
  extractor = EmailFeatureExtractor(min_df=1)

11.4 TEST FAILURES
------------------
Problem: Tests fail with "Resource not found"
Solution: Download NLTK data before running tests

Problem: Tests fail with permission errors on Windows
Solution: Run terminal as Administrator or use different temp directory

Problem: Tests hang or timeout
Solution: Reduce test data size or check for infinite loops in preprocessing

11.5 SAVE/LOAD ERRORS
---------------------
Problem: FileNotFoundError when saving artifacts
Solution: Ensure artifacts directory exists:
  New-Item -ItemType Directory -Force -Path "artifacts\models"

Problem: PermissionError when loading artifacts
Solution: Close any programs that might have the file open, or use different path

Problem: Loaded extractor produces different features
Solution: Ensure scikit-learn version is same across save/load sessions

11.6 PERFORMANCE ISSUES
-----------------------
Problem: Preprocessing very slow on large datasets
Solution: Disable lemmatization or reduce features:
  TextPreprocessor(apply_lemmatization=False)

Problem: Memory error during vectorization
Solution: Reduce max_features or process in batches:
  EmailFeatureExtractor(max_features=1000)

11.7 FEATURE MISMATCH ERRORS
----------------------------
Problem: ValueError: X has different number of features than during training
Solution: This is CRITICAL error. Causes:
  - Using different vectorizer for training vs inference
  - Vectorizer not fitted before transform
  - Using new vectorizer instead of loading trained one

Fix: ALWAYS save and load the SAME vectorizer:
  # Training
  extractor.fit(train_texts)
  extractor.save("vectorizer.pkl")
  
  # Inference
  extractor = EmailFeatureExtractor.load("vectorizer.pkl")  # Use SAME vectorizer

11.8 GETTING HELP
-----------------
If issues persist:
1. Check Python version (should be 3.8+)
2. Verify virtual environment is activated
3. Check all dependencies installed correctly
4. Review error messages carefully
5. Check file paths are correct (absolute vs relative)
6. Ensure working directory is project root


12.1 DATA LOADING ERRORS
-------------------------
Problem: FileNotFoundError when loading data
Solution: Check file path is correct and file exists
  dir data\sample_emails.csv

Problem: "Missing required columns" error
Solution: CSV must have 'text' and 'category' columns
  Check CSV header row matches expected format

Problem: UnicodeDecodeError when loading CSV
Solution: Specify encoding explicitly:
  loader.load_csv('data/file.csv', encoding='utf-8-sig')

12.2 TRAINING ERRORS
--------------------
Problem: "ValueError: empty vocabulary"
Solution: Not enough unique words. Solutions:
  - Use more training data
  - Reduce min_df: --max-features 1000
  - Check data isn't all duplicates

Problem: "ValueError: y contains X classes, but estimator expects Y"
Solution: Mismatch between training and test labels
  - Ensure test set has same categories as training
  - Check for typos in category names

Problem: Training very slow (hours)
Solution: 
  - Disable hyperparameter tuning: --no-tune
  - Reduce features: --max-features 1000
  - Use simpler model: --model naive_bayes

Problem: Memory error during training
Solution:
  - Reduce max_features
  - Use smaller dataset
  - Close other applications

12.3 EVALUATION ERRORS
----------------------
Problem: Confusion matrix image not saved
Solution: Check matplotlib backend:
  matplotlib.use('Agg') should be set in evaluator.py

Problem: Division by zero in metrics
Solution: Some classes have zero samples in test set
  - Use stratified splitting (already default)
  - Increase test set size

12.4 DATABASE ERRORS
--------------------
Problem: "Table 'model_versions' doesn't exist"
Solution: Step 1 schema not created. Run:
  Execute create_schema.sql in Neon SQL Editor

Problem: "Connection refused" to database
Solution: Check .env credentials:
  - DB_HOST correct
  - DB_USER correct
  - DB_PASSWORD correct
  - DB_SSLMODE=require

Problem: "Duplicate key value" when registering model
Solution: Model version already exists
  - Use different version name: --version 1.1
  - Or delete old model from database

12.5 ARTIFACT ERRORS
--------------------
Problem: "No such file or directory" when saving artifacts
Solution: Create directories:
  New-Item -ItemType Directory -Force -Path "artifacts\models"
  New-Item -ItemType Directory -Force -Path "artifacts\reports"

Problem: Loaded model produces different predictions
Solution: 
  - Ensure same scikit-learn version
  - Check model file not corrupted
  - Verify vectorizer loaded correctly

12.6 POOR MODEL PERFORMANCE
---------------------------
Problem: Accuracy < 50% (worse than random)
Solution: Possible causes:
  - Data quality issues (labels incorrect)
  - Too few training samples per category
  - Categories too similar (hard to distinguish)
  - Preprocessing too aggressive (losing information)

Fixes:
  - Review and clean labels
  - Collect more training data
  - Try different model: --model random_forest
  - Reduce preprocessing (edit pipeline config)

Problem: High training accuracy, low test accuracy (overfitting)
Solution:
  - Use regularization (already in Logistic Regression)
  - Reduce max_features
  - Collect more training data
  - Use cross-validation (already done)

Problem: All predictions are same class
Solution: Severe class imbalance
  - Check class distribution (should be balanced)
  - Use class weights (already enabled with 'balanced')
  - Oversample minority class with SMOTE

12.7 GETTING HELP
-----------------
If issues persist:
1. Check all Step 1 & 2 components working
2. Verify sample data format matches expected
3. Review error messages carefully
4. Check Python version (3.8+)
5. Ensure virtual environment activated
6. Try simpler configuration first (--no-tune, fewer features)


13.1 MODEL LOADING ERRORS
--------------------------
Problem: "No active model found in database"
Solution:
1. Check if any models exist:
   python -c "from training import ModelRegistry; print(len(ModelRegistry().list_all_models()))"

2. If 0, train a model:
   python train_model.py --data data/sample_emails.csv --version 1.0 --set-active

3. If >0 but none active, activate one:
   python -c "from training import ModelRegistry; ModelRegistry().set_model_active(1)"

Problem: "Model file not found"
Solution:
1. Check model path in database:
   python -c "from training import ModelRegistry; m = ModelRegistry().get_active_model(); print(m.model_path)"

2. Verify file exists:
   Test-Path "path\from\above"

3. If missing, retrain model:
   python train_model.py --data data/sample_emails.csv --version 1.0 --set-active

13.2 PREDICTION ERRORS
----------------------
Problem: All predictions return "unknown"
Cause: Empty or invalid input
Solution: Check input text is valid:
  - Not empty
  - Is a string
  - Has actual content

Problem: Low confidence on all predictions
Cause: Model not trained well or input very different from training data
Solution:
1. Check model accuracy:
   python -c "from training import ModelRegistry; m = ModelRegistry().get_active_model(); print(f'Accuracy: {m.accuracy}')"

2. If low (<70%), retrain with more data
3. If high but predictions still low confidence, input may be out-of-domain

Problem: Predictions inconsistent
Cause: Different preprocessing between training and inference
Solution: This shouldn't happen (same pipeline used), but verify:
  - Same model file used
  - Same vectorizer used
  - No manual text modification before prediction

13.3 DATABASE ERRORS
--------------------
Problem: "Table 'predictions' doesn't exist"
Solution: Create schema (from Step 1):
  - Log into Neon dashboard
  - Run create_schema.sql in SQL Editor

Problem: Foreign key constraint error
Cause: email_id doesn't exist in emails table
Solution: Create email first:
  from database.repository import EmailRepository
  email = EmailRepository().create_email(subject="", body="text", sender="user@example.com")
  # Then use email.id in save_prediction()

13.4 PERFORMANCE ISSUES
-----------------------
Problem: Predictions very slow (>1 second each)
Cause: Model loading on every prediction
Solution: 
  - EmailCategorizer should cache model (default)
  - Don't create new EmailCategorizer for each prediction
  - Reuse same instance:
    categorizer = EmailCategorizer()  # Once
    categorizer.predict(email1)        # Fast
    categorizer.predict(email2)        # Fast

Problem: Batch predictions slow
Cause: Processing one at a time instead of batch
Solution: Use predict_batch():
  results = categorizer.predict_batch(emails)  # Good
  # Not: [categorizer.predict(e) for e in emails]  # Bad

13.5 CONFIDENCE ISSUES
----------------------
Problem: All confidences very low (<0.5)
Cause: Model uncertainty or poor training
Solution:
1. Review training data quality
2. Retrain with more data
3. Try different algorithm
4. Check if input domain matches training domain

Problem: All confidences very high (>0.95)
Cause: Model overconfident (might be overfitting)
Solution:
1. Check test set accuracy from training
2. If train accuracy >> test accuracy, model overfitting
3. Retrain with regularization or simpler model

13.6 GETTING HELP
-----------------
If issues persist:
1. Check all Steps 1-3 working correctly
2. Verify active model exists and loads
3. Test with simple, clear examples first
4. Review error messages carefully
5. Check database connection
6. Ensure working directory is project root