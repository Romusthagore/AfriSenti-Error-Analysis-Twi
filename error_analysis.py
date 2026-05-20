# error_analysis.py
# Error analysis for Twi sentiment model (Seed 2)
# Location: /content/drive/MyDrive/AfriSenti/error_analysis.py

from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np
import emoji
import re

def load_model_and_data():
    """Load model from Drive and prepare dataset."""
    model_path = "/content/drive/MyDrive/AfriSenti/models/twi_baseline_seed2"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model loaded from Drive")
    
    tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")
    dataset = load_from_disk("/content/drive/MyDrive/AfriSenti/data/twi")
    
    def tokenize(x):
        return tokenizer(x["tweet"], padding="max_length", truncation=True, max_length=200)
    
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    return model, dataset

def get_predictions(model, dataset):
    """Get predictions on validation set."""
    trainer = Trainer(model=model)
    preds = trainer.predict(dataset["validation"])
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids
    return y_true, y_pred

def find_errors(y_true, y_pred):
    """Find misclassified indices."""
    error_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    print(f"\nErrors: {len(error_indices)}/{len(y_true)} ({len(error_indices)/len(y_true):.1%})")
    return error_indices

def classification_report_validation(y_true, y_pred):
    """Print classification report."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT - VALIDATION SET")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=["pos", "neu", "neg"]))

def display_sample_errors(error_indices, y_true, y_pred):
    """Display first 15 misclassified examples."""
    dataset_raw = load_from_disk("/content/drive/MyDrive/AfriSenti/data/twi")
    label_names = ["positive", "neutral", "negative"]
    
    print("\n" + "="*60)
    print("SAMPLE ERRORS (first 15)")
    print("="*60)
    
    for i in error_indices[:15]:
        true_label = label_names[y_true[i]]
        pred_label = label_names[y_pred[i]]
        print(f"\nTrue: {true_label:8} | Pred: {pred_label:8}")
        print(f"Text: {dataset_raw['validation'][i]['tweet'][:200]}")

def confusion_summary(error_indices, y_true, y_pred):
    """Print confusion summary."""
    dataset_raw = load_from_disk("/content/drive/MyDrive/AfriSenti/data/twi")
    label_names = ["positive", "neutral", "negative"]
    
    print("\n" + "="*60)
    print("CONFUSION SUMMARY")
    print("="*60)
    
    confusions = [(label_names[y_true[i]], label_names[y_pred[i]]) for i in error_indices]
    for (true, pred), count in Counter(confusions).most_common(10):
        print(f"  {true} → {pred}: {count} times")

def pattern_analysis(error_indices):
    """Analyze patterns in errors."""
    dataset_raw = load_from_disk("/content/drive/MyDrive/AfriSenti/data/twi")
    error_texts = [dataset_raw['validation'][i]['tweet'] for i in error_indices]
    
    emoji_count = sum(1 for t in error_texts if any(c in emoji.EMOJI_DATA for c in t))
    exclamation_count = sum(1 for t in error_texts if '!' in t)
    question_count = sum(1 for t in error_texts if '?' in t)
    repeat_count = sum(1 for t in error_texts if re.search(r'(.)\1{2,}', t))
    
    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)
    print(f"\nPattern frequency in {len(error_texts)} errors:")
    print(f"   Emojis        : {emoji_count}/{len(error_texts)} ({emoji_count/len(error_texts)*100:.1f}%)")
    print(f"   Exclamation   : {exclamation_count}/{len(error_texts)} ({exclamation_count/len(error_texts)*100:.1f}%)")
    print(f"   Question      : {question_count}/{len(error_texts)} ({question_count/len(error_texts)*100:.1f}%)")
    print(f"   Repeated chars: {repeat_count}/{len(error_texts)} ({repeat_count/len(error_texts)*100:.1f}%)")

def evaluate_test_set(model, dataset):
    """Evaluate model on test set."""
    trainer = Trainer(model=model)
    test_preds = trainer.predict(dataset["test"])
    y_test_pred = np.argmax(test_preds.predictions, axis=1)
    y_test_true = test_preds.label_ids
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(classification_report(y_test_true, y_test_pred, target_names=["pos", "neu", "neg"]))

def main():
    """Run complete error analysis."""
    model, dataset = load_model_and_data()
    y_true, y_pred = get_predictions(model, dataset)
    error_indices = find_errors(y_true, y_pred)
    classification_report_validation(y_true, y_pred)
    display_sample_errors(error_indices, y_true, y_pred)
    confusion_summary(error_indices, y_true, y_pred)
    pattern_analysis(error_indices)
    evaluate_test_set(model, dataset)

if __name__ == "__main__":
    main()

print("\n✅ File created at: /content/drive/MyDrive/AfriSenti/error_analysis.py")
