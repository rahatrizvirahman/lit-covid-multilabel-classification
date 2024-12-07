

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
from tqdm import tqdm
import os
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import os
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import time

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_roc_curves(y_true, y_pred, labels, output_dir):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, 
            tpr, 
            label=f'{label} (AUC = {roc_auc:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def get_metric(self, metric_name):
        return self.metrics[metric_name]

def plot_training_history(tracker, fold, output_dir):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(tracker.get_metric('train_loss'), label='Train Loss')
    plt.plot(tracker.get_metric('val_loss'), label='Validation Loss')
    plt.title(f'Loss History - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(tracker.get_metric('exact_match_accuracy'), label='Exact Match')
    plt.plot(tracker.get_metric('hamming_accuracy'), label='Hamming')
    plt.title(f'Accuracy History - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1, Precision, Recall
    plt.subplot(2, 2, 3)
    plt.plot(tracker.get_metric('f1'), label='F1')
    plt.plot(tracker.get_metric('precision'), label='Precision')
    plt.plot(tracker.get_metric('recall'), label='Recall')
    plt.title(f'Metrics History - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_history_fold_{fold}.png'))
    plt.close()
    
def plot_confusion_matrices(y_true, y_pred, labels, output_dir):
    """Plot confusion matrix for each class"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    n_classes = len(labels)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, label in enumerate(labels):
        cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix - {label}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    if len(labels) < len(axes):
        for idx in range(len(labels), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

def plot_label_distribution(train_labels, test_labels, labels, output_dir):
    """Plot label distribution in train and test sets"""
    train_dist = train_labels.sum(axis=0)
    test_dist = test_labels.sum(axis=0)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, train_dist, width, label='Train')
    plt.bar(x + width/2, test_dist, width, label='Test')
    
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Label Distribution in Train and Test Sets')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.close()

def create_performance_tables(y_true, y_pred, labels, output_dir):
    """Create and save detailed performance tables"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    metrics_dict = {
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'Support': []
    }
    
    for i in range(len(labels)):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, i], y_pred_binary[:, i], average='binary'
        )
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1-Score'].append(f1)
        metrics_dict['Support'].append(support)
    
    df_metrics = pd.DataFrame(metrics_dict, index=labels)
    df_metrics.to_csv(os.path.join(output_dir, 'class_performance_metrics.csv'))
    
    corr_matrix = np.corrcoef(y_pred_binary.T)
    df_corr = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    df_corr.to_csv(os.path.join(output_dir, 'prediction_correlations.csv'))
    
    return df_metrics, df_corr


class DocumentProcessor:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Initialize labels as None - will be set later
        self.labels = None
        
    def get_unique_labels(self, train_df):
        """Extract unique labels from training data"""
        # Get all unique labels by splitting the semicolon-separated values
        all_labels = set()
        for labels in train_df['label']:
            all_labels.update(labels.split(';'))
        # Sort for consistency
        self.labels = sorted(list(all_labels))
        print(f"Found {len(self.labels)} unique labels: {self.labels}")
        return self.labels
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = ' '.join(text.split())
        return text
    
    def process_labels(self, label_text):
        if self.labels is None:
            raise ValueError("Labels have not been initialized. Call get_unique_labels first.")
            
        label_list = label_text.split(';')
        label_array = np.zeros(len(self.labels))
        for label in label_list:
            if label in self.labels:
                label_array[self.labels.index(label)] = 1
            else:
                print(f"Warning: Unknown label encountered: {label}")
        return label_array
    
    def tokenize_text(self, text):
        """Tokenize text using BERT tokenizer"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding
    

class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=7):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        # Define dimensions
        self.bert_hidden_size = self.bert.config.hidden_size  # 768
        self.hidden_size1 = 512
        self.hidden_size2 = 256
        
        # Layers
        self.layer1 = nn.Linear(self.bert_hidden_size, self.hidden_size1)
        self.layer2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.classifier = nn.Linear(self.hidden_size2, num_labels)
        
        # Activation and regularization
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()  # Added this line - initialize ReLU
        
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        
        # Additional layers with activation and dropout
        x = self.dropout(pooled_output)
        x = self.relu(self.layer1(x))
        
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return torch.sigmoid(logits)

def plot_metrics_heatmap(metrics_dict, labels, output_dir):
    """Create a heatmap of metrics for each category"""
    metrics_df = pd.DataFrame({
        'Precision': metrics_dict['Precision'],
        'Recall': metrics_dict['Recall'],
        'F1-Score': metrics_dict['F1-Score']
    }, index=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Performance Metrics by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
    plt.close()
    
def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for multi-label classification"""
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Exact match accuracy (all labels must match)
    exact_match_accuracy = np.mean(np.all(y_pred_binary == y_true, axis=1))
    
    # Per-class accuracy
    per_class_accuracy = np.mean(y_pred_binary == y_true, axis=0)
    
    # Hamming accuracy (proportion of correct predictions)
    hamming_accuracy = np.mean(y_pred_binary == y_true)
    
    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='samples'
    )
    
    return {
        'exact_match_accuracy': exact_match_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_overall_metrics(y_true, y_pred, labels, output_dir):
    """Calculate both overall and per-category metrics"""
    # Verify input dimensions
    assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
    assert y_true.shape[1] == len(labels), f"Number of labels mismatch: {y_true.shape[1]} != {len(labels)}"
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Per-category metrics
    per_category_metrics = {
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'Support': []
    }
    
    print("\nPer-category Metrics:")
    print("--------------------")
    for i, label in enumerate(labels):
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true[:, i], y_pred_binary[:, i], average='binary'
            )
            per_category_metrics['Precision'].append(precision)
            per_category_metrics['Recall'].append(recall)
            per_category_metrics['F1-Score'].append(f1)
            per_category_metrics['Support'].append(support)
            
            print(f"\n{label}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Support: {support}")
            
        except Exception as e:
            print(f"Error processing label {label} at index {i}: {str(e)}")
            print(f"Label shape: {y_true[:, i].shape}")
            print(f"Prediction shape: {y_pred_binary[:, i].shape}")
            raise
    
    # Overall metrics
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='micro'
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='weighted'
    )
    
    
    # Exact match ratio (perfect predictions across all categories)
    exact_match = np.mean(np.all(y_pred_binary == y_true, axis=1))
    
    # Hamming accuracy (percentage of correct labels)
    hamming_accuracy = np.mean(y_pred_binary == y_true)
    
    # Create summary dictionary
    overall_metrics = {
        'Micro-average': {
            'Precision': micro_precision,
            'Recall': micro_recall,
            'F1-Score': micro_f1
        },
        'Macro-average': {
            'Precision': macro_precision,
            'Recall': macro_recall,
            'F1-Score': macro_f1
        },
        'Weighted-average': {
            'Precision': weighted_precision,
            'Recall': weighted_recall,
            'F1-Score': weighted_f1
        },
        'Exact Match Ratio': exact_match,
        'Hamming Accuracy': hamming_accuracy
    }
    
    # Create and display summary DataFrame
    df_overall = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Micro-avg': [micro_precision, micro_recall, micro_f1],
        'Macro-avg': [macro_precision, macro_recall, macro_f1],
        'Weighted-avg': [weighted_precision, weighted_recall, weighted_f1]
    }).set_index('Metric')
    
    print("\nOverall Metrics:")
    print("--------------")
    print(f"\nExact Match Ratio: {exact_match:.4f}")
    print(f"Hamming Accuracy: {hamming_accuracy:.4f}")
    print("\nAveraged Metrics:")
    print(df_overall)
    
    # Save metrics to CSV
    df_overall.to_csv(os.path.join(output_dir,'overall_metrics.csv'))
    df_categories = pd.DataFrame(per_category_metrics, index=labels)
    df_categories.to_csv(os.path.join(output_dir,'per_category_metrics.csv'))
    
    return overall_metrics, per_category_metrics

class BERTTrainer:
    def __init__(self, model, device, output_dir):
        self.model = model.to(device)  # Move model to device immediately
        self.device = device
        self.output_dir = output_dir
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        self.tracker = MetricTracker()
        
    
    def evaluate(self, loader):
        """Evaluate model during training"""
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                # Move batch tensors to device
                input_ids = batch['input_ids'].to(trainer.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = outputs.cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        metrics = calculate_metrics(all_labels, all_predictions)
        
        return val_loss/len(loader), metrics
    
    def evaluate_final(self, loader, labels):
        """Final evaluation on test set"""
        self.model.eval()
        test_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, batch_labels)
                test_loss += loss.item()

                # Move predictions and labels to CPU and convert to numpy
                predictions = outputs.cpu().numpy()
                labels_np = batch_labels.cpu().numpy()

                all_predictions.append(predictions)
                all_labels.append(labels_np)

        # Concatenate all batches
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        # Ensure shapes match
        assert all_predictions.shape == all_labels.shape, \
            f"Shape mismatch: predictions {all_predictions.shape} != labels {all_labels.shape}"

        overall_metrics, per_category_metrics = calculate_overall_metrics(
            all_labels, all_predictions, labels, self.output_dir
        )

        return test_loss/len(loader), all_predictions, all_labels, overall_metrics, per_category_metrics


def process_data(df, processor):
    """
    Process dataframe into BERT-ready dataset
    
    Args:
        df: pandas DataFrame containing 'abstract' and 'label' columns
        processor: DocumentProcessor instance
    
    Returns:
        COVIDDataset instance
    """
    # Clean abstracts
    abstracts = df['abstract'].apply(processor.clean_text).values
    
    # Convert labels to multi-hot encoding
    labels = np.array([processor.process_labels(label) for label in df['label']])
    
    # Tokenize texts
    encodings = []
    for abstract in tqdm(abstracts, desc="Tokenizing texts"):
        encoding = processor.tokenize_text(abstract)
        encodings.append({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        })
    
    # Create dataset
    dataset = COVIDDataset(
        texts=abstracts,
        labels=labels,
        processor=processor
    )
    
    print(f"Processed {len(dataset)} samples")
    return dataset

class COVIDDataset(Dataset):
    def __init__(self, texts, labels, processor):
        """
        Args:
            texts: list of abstract texts
            labels: numpy array of multi-hot encoded labels
            processor: DocumentProcessor instance
        """
        self.texts = texts
        self.labels = torch.FloatTensor(labels)
        self.processor = processor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.processor.tokenize_text(text)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': self.labels[idx]
        }


if __name__ == "__main__":
    output_dir = 'output/bert'
    ensure_output_dir(output_dir)
    model_name = 'bert-base-uncased'

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv('./dataset/BC7-LitCovid-Train.csv')
    val_df = pd.read_csv('./dataset/BC7-LitCovid-Dev.csv')
    test_df = pd.read_csv('./dataset/BC7-LitCovid-Test-GS.csv')

    # Initialize processor
    processor = DocumentProcessor(model_name=model_name)

    # Initialize labels using training data
    processor.get_unique_labels(train_df)


    train_labels = np.array([processor.process_labels(label) for label in train_df['label']])



    # Process data
    print("Processing datasets...")
    train_dataset = process_data(train_df, processor)
    val_dataset = process_data(val_df, processor)
    test_dataset = process_data(test_df, processor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTClassifier(model_name=model_name, num_labels=len(processor.labels))

    # Initialize trainer and run training
    trainer = BERTTrainer(model, device, output_dir)
    total_epochs=5

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    training_start_time = time.time()

    # Training loop
    for epoch in range(total_epochs):
        # Training phase
        trainer.model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} - Training'):
            # Move all batch tensors to device
            input_ids = batch['input_ids'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            labels = batch['labels'].to(trainer.device)

            trainer.optimizer.zero_grad()
            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss, val_metrics = trainer.evaluate(val_loader)
        avg_train_loss = train_loss/len(train_loader)

        # Track metrics
        trainer.tracker.update({
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **val_metrics
        })

        # Print metrics
        print(f'Epoch {epoch+1}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print('Validation Metrics:')
        print(f'  Exact Match Accuracy: {val_metrics["exact_match_accuracy"]:.4f}')
        print(f'  Hamming Accuracy: {val_metrics["hamming_accuracy"]:.4f}')
        print(f'  F1: {val_metrics["f1"]:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(trainer.model.state_dict(), os.path.join(trainer.output_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    training_end_time = time.time()

    print(f"Training time: {training_end_time - training_start_time:.2f} seconds")

    # Plot training history
    plot_training_history(trainer.tracker, 1, trainer.output_dir)

    # Final evaluation on test set
    print("\nEvaluating on test set...")

    evaluation_start_time = time.time()

    trainer.model.load_state_dict(torch.load(os.path.join(trainer.output_dir, 'best_model.pt')))
    test_loss, test_predictions, test_labels, overall_metrics, per_category_metrics = trainer.evaluate_final(
        test_loader, processor.labels
    )

    evaluation_end_time = time.time()


    # Create visualizations
    plot_confusion_matrices(test_labels, test_predictions, processor.labels, output_dir)
    plot_roc_curves(test_labels, test_predictions, processor.labels, output_dir)
    plot_metrics_heatmap(per_category_metrics, processor.labels, output_dir)
    plot_label_distribution(train_labels, test_labels, processor.labels, output_dir)

    # Save detailed performance analysis
    create_performance_tables(test_labels, test_predictions, processor.labels, output_dir)

    # Save summary metrics
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Overall Metrics:\n")
        f.write("---------------\n")
        for metric_type, metrics in overall_metrics.items():
            f.write(f"\n{metric_type}:\n")
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{metrics:.4f}\n")


