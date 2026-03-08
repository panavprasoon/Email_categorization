How to use the training pipeline:

# Basic training (auto version, logistic regression)
# Trains the entire pipeline, validate that everything works and create a complete trained model 
# Saves all artifacts and save the registers in db 
# generated reports and confusion matrix
# uses logistic regression [speed == fast, acurracy == good, use when want a baseline and interpretable model]
python train_model.py --data data/sample_emails.csv


# Train specific model type
# can use random forest[speed == slow, acurracy == better, use when there are complex paaterns and some features with importance]
# can use svm[speed == slow, acurracy == good, use when there is text classifcation and its good with small data]
# can use naive bayes[speed == very fast, acurracy == decent, when there is limited data and we need quick testing]
python train_model.py --data data/sample_emails.csv --model random_forest

# Custom version and set as active
# When deploying to production
# can have custom version names
python train_model.py --data data/sample_emails.csv --version 1.0 --set-active

# Train without hyperparameter tuning (faster)
# for quick testing and with small dataset as there won't be any tuning
# There is a time constraint
python train_model.py --data data/sample_emails.csv --no-tune

# Custom data split ratios
python train_model.py --data data/sample_emails.csv --test-size 0.3 --val-size 0.15

# Reduce features for faster training
# max feature                 Pros                       Cons
# 500[ for wuick testing]     [very fast]                [ lower accuracy]         
# 1000[ resource oriented]    [fast due to small files]  [ some accuracy loss]
# 5000[ balanced]             [good speed/accuracy]      [ moderate resources]
# 10000[ for larger dataset]  [high accuracy]            [ slower]
# 50000[ for maximum detail]  [best accuracy]            [ very slow due to huge files]
python train_model.py --data data/sample_emails.csv --max-features 1000

# Full example with all options
python train_model.py \
    --data data/sample_emails.csv \
    --version 1.0 \
    --model logistic_regression \
    --test-size 0.2 \
    --val-size 0.1 \
    --max-features 5000 \
    --set-active \
    --description "Production model v1.0"
