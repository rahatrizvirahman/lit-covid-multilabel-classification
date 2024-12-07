

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
import time


output_dir = 'output/sbert'

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
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.labels = ['Treatment', 'Prevention', 'Diagnosis', 'Mechanism', 
                      'Transmission', 'Epidemic Forecasting', 'Case Report']
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = ' '.join(text.split())
        return text
    
    def process_labels(self, label_text):
        label_list = label_text.split(';')
        label_array = np.zeros(len(self.labels))
        for label in label_list:
            if label in self.labels:
                label_array[self.labels.index(label)] = 1
        return label_array
    
    def generate_embeddings(self, texts, batch_size=32, cache_file=None):
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Generating new embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        if cache_file:
            print(f"Caching embeddings to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings

class COVIDDataset(Dataset):
    def __init__(self, embeddings, labels):
        assert len(embeddings) == len(labels), "Embeddings and labels must have same length"
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
        print(f"Dataset size: {len(self.embeddings)} samples") 
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if idx >= len(self.embeddings):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.embeddings)}")
        return self.embeddings[idx], self.labels[idx]

class TopicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, num_classes=7):
        super(TopicClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.layer3(x))
        return x

# Modify train_fold function to use MetricTracker
def train_fold(model, train_loader, val_loader, criterion, optimizer, device, fold, output_dir):
    model = model.to(device)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    tracker = MetricTracker()
    
    for epoch in range(20):
        # Training phase
        model.train()
        train_loss = 0
        for batch_embeddings, batch_labels in tqdm(train_loader, desc=f'Fold {fold}, Epoch {epoch+1} - Training'):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        val_loss, val_metrics = evaluate_fold(model, val_loader, criterion, device, fold)
        
        avg_train_loss = train_loss/len(train_loader)
        
        # Track metrics
        tracker.update({
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **val_metrics
        })
        
        print(f'Fold {fold}, Epoch {epoch+1}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print('Validation Metrics:')
        print(f'  Exact Match Accuracy: {val_metrics["exact_match_accuracy"]:.4f}')
        print(f'  Hamming Accuracy: {val_metrics["hamming_accuracy"]:.4f}')
        print(f'  F1: {val_metrics["f1"]:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_fold_{fold}.pt'))

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Plot training history for this fold
    plot_training_history(tracker, fold, output_dir)
    return best_val_loss, val_metrics


def evaluate_model(model, test_loader, criterion, device, labels, output_dir):
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in tqdm(test_loader, desc='Testing'):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            
            predictions = outputs.cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    overall_metrics, per_category_metrics = calculate_overall_metrics(
        all_labels, all_predictions, labels, output_dir
    )
    
    return test_loss/len(test_loader), all_predictions, all_labels, overall_metrics, per_category_metrics

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
    """
    Calculate both overall and per-category metrics
    """
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
    
    # Overall metrics (micro average)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='micro'
    )
    
    # Overall metrics (macro average)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='macro'
    )
    
    # Overall metrics (weighted average)
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

def evaluate_fold(model, val_loader, criterion, device, fold):
    """Evaluate model on validation set during training"""
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in val_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            
            predictions = outputs.cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics with binary predictions
    metrics = calculate_metrics(all_labels, all_predictions)
    
    return val_loss/len(val_loader), metrics

if __name__ == "__main__":
    model_name='all-mpnet-base-v2'
    ensure_output_dir(output_dir)

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv('./dataset/BC7-LitCovid-Train.csv')
    val_df = pd.read_csv('./dataset/BC7-LitCovid-Dev.csv')  
    test_df = pd.read_csv('./dataset/BC7-LitCovid-Test-GS.csv')

    # Initialize processor
    processor = DocumentProcessor(model_name)

    # Process training data
    print("Processing training data...")
    train_abstracts = train_df['abstract'].apply(processor.clean_text).values
    train_labels = np.array([processor.process_labels(label) for label in train_df['label']])
    train_embeddings = processor.generate_embeddings(train_abstracts, cache_file=f'{output_dir}/train_embeddings_cache.pkl')

    # Process validation data
    print("Processing validation data...")
    val_abstracts = val_df['abstract'].apply(processor.clean_text).values
    val_labels = np.array([processor.process_labels(label) for label in val_df['label']])
    val_embeddings = processor.generate_embeddings(val_abstracts, cache_file=f'{output_dir}/val_embeddings_cache.pkl')

    # Process test data
    print("Processing test data...")
    test_abstracts = test_df['abstract'].apply(processor.clean_text).values
    test_labels = np.array([processor.process_labels(label) for label in test_df['label']])
    test_embeddings = processor.generate_embeddings(test_abstracts, cache_file=f'{output_dir}/test_embeddings_cache.pkl')

    # Get embedding dimension
    embedding_dim = train_embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    # Create datasets
    train_dataset = COVIDDataset(train_embeddings, train_labels)
    val_dataset = COVIDDataset(val_embeddings, val_labels)
    test_dataset = COVIDDataset(test_embeddings, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TopicClassifier(input_dim=embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("\nTraining model...")
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    tracker = MetricTracker()

    training_start_time = time.time()

    total_epoch=5

    for epoch in range(total_epoch):
        # Training phase
        model.train()
        train_loss = 0
        for batch_embeddings, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1} - Training'):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss, val_metrics = evaluate_fold(model, val_loader, criterion, device, 1)

        avg_train_loss = train_loss/len(train_loader)

        # Track metrics
        tracker.update({
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **val_metrics
        })

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
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    training_end_time = time.time()

    # Plot training history
    plot_training_history(tracker, 1, output_dir)

    evaluation_start_time = time.time()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    test_loss, test_predictions, test_labels_array, overall_metrics, per_category_metrics = evaluate_model(
        model, test_loader, criterion, device, processor.labels, output_dir
    )

    evaluation_end_time = time.time()

    # Create visualizations
    plot_confusion_matrices(test_labels_array, test_predictions, processor.labels, output_dir)
    plot_roc_curves(test_labels_array, test_predictions, processor.labels, output_dir)
    plot_metrics_heatmap(per_category_metrics, processor.labels, output_dir)
    plot_label_distribution(train_labels, test_labels_array, processor.labels, output_dir)

    # Save detailed performance analysis
    create_performance_tables(test_labels_array, test_predictions, processor.labels, output_dir)

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
