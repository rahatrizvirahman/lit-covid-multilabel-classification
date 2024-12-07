

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
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

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
    
    # Plot metrics
    plt.subplot(2, 2, 2)
    if 'accuracy' in tracker.metrics:
        plt.plot(tracker.get_metric('accuracy'), label='Accuracy')
    if 'f1' in tracker.metrics:
        plt.plot(tracker.get_metric('f1'), label='F1')
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
    def __init__(self, embedding_dim=300, max_length=200):
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.word2vec = None
        self.word2idx = {}
        self.vocab_size = 0
        self.labels = None
        
    def load_word2vec(self, texts):
        """Train Word2Vec on our corpus"""
        # Tokenize all texts
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Train Word2Vec
        self.word2vec = Word2Vec(sentences=tokenized_texts, 
                                vector_size=self.embedding_dim, 
                                window=5, 
                                min_count=1, 
                                workers=4)
        
        # Create word to index mapping
        self.word2idx = {word: idx + 1 for idx, word in 
                        enumerate(self.word2vec.wv.index_to_key)}
        self.vocab_size = len(self.word2idx) + 1  # +1 for padding
        
        # Create embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, idx in self.word2idx.items():
            self.embedding_matrix[idx] = self.word2vec.wv[word]
    
    def get_unique_labels(self, train_df):
        all_labels = set()
        for labels in train_df['label']:
            all_labels.update(labels.split(';'))
        self.labels = sorted(list(all_labels))
        return self.labels
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        return str(text).lower()
    
    def process_labels(self, label_text):
        label_list = label_text.split(';')
        label_array = np.zeros(len(self.labels))
        for label in label_list:
            if label in self.labels:
                label_array[self.labels.index(label)] = 1
        return label_array
    
    def text_to_sequence(self, text):
        """Convert text to sequence of word indices"""
        tokens = word_tokenize(text.lower())
        sequence = [self.word2idx.get(token, 0) for token in tokens[:self.max_length]]
        # Pad sequence
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        return torch.LongTensor(sequence)
    

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1=512, hidden_dim2=256, num_classes=7):
        super(BiLSTMClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim1, 
                           num_layers=2, 
                           bidirectional=True, 
                           batch_first=True,
                           dropout=0.2)
        
        # Additional layers
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim1)  # *2 for bidirectional
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)
    
    def forward(self, x):
        # Get embeddings
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Get final hidden state for both directions
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        
        # Additional layers with residual connections
        x = self.dropout(lstm_out)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)

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
    
def evaluate_final(model, test_loader, labels, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            sequences = batch['sequence'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(sequences)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            predictions = outputs.cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    overall_metrics, per_category_metrics = calculate_overall_metrics(
        all_labels, all_predictions, labels, output_dir
    )

    return test_loss/len(test_loader), all_predictions, all_labels, overall_metrics, per_category_metrics

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metrics_dict):
        """Update metrics with a dictionary of values"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def get_metric(self, metric_name):
        """Get the list of values for a specific metric"""
        return self.metrics[metric_name]
        
def plot_training_curves(tracker, output_dir):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(tracker.train_losses, label='Train Loss')
    plt.plot(tracker.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

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
        self.texts = texts
        self.labels = torch.FloatTensor(labels)
        self.processor = processor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        sequence = self.processor.text_to_sequence(text)
        return {
            'sequence': sequence,
            'labels': self.labels[idx]
        }
    
if __name__ == "__main__":
    output_dir = 'output/lstm'
    ensure_output_dir(output_dir)

    metrics_tracker = MetricsTracker()

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv('./dataset/BC7-LitCovid-Train.csv')
    val_df = pd.read_csv('./dataset/BC7-LitCovid-Dev.csv')
    test_df = pd.read_csv('./dataset/BC7-LitCovid-Test-GS.csv')


    # Initialize processor
    processor = DocumentProcessor(embedding_dim=300, max_length=200)
    processor.get_unique_labels(train_df)

    # Clean texts
    train_texts = train_df['abstract'].apply(processor.clean_text).values
    val_texts = val_df['abstract'].apply(processor.clean_text).values
    test_texts = test_df['abstract'].apply(processor.clean_text).values

    # Train Word2Vec and create embeddings
    print("Training Word2Vec model...")
    processor.load_word2vec(train_texts)

    # Process labels
    train_labels = np.array([processor.process_labels(label) for label in train_df['label']])
    val_labels = np.array([processor.process_labels(label) for label in val_df['label']])
    test_labels = np.array([processor.process_labels(label) for label in test_df['label']])


    # Create datasets
    train_dataset = COVIDDataset(train_texts, train_labels, processor)
    val_dataset = COVIDDataset(val_texts, val_labels, processor)
    test_dataset = COVIDDataset(test_texts, val_labels, processor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = BiLSTMClassifier(
        vocab_size=processor.vocab_size,
        embedding_dim=processor.embedding_dim,
        num_classes=len(processor.labels)
    )

    # Initialize embedding layer with pretrained embeddings
    model.embedding.weight.data.copy_(torch.from_numpy(processor.embedding_matrix))

    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_model(model, train_loader, val_loader, num_epochs=5, device=device)

    total_epoch = 5

    training_start_time = time.time()


    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_val_loss = float('inf')
    # Replace the main training section with:
    for epoch in range(total_epoch):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
            sequences = batch['sequence'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sequences = batch['sequence'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)
        val_metrics = calculate_metrics(val_true_labels, val_predictions)
        
        # Update metrics tracker
        metrics_tracker.update({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': val_metrics['exact_match_accuracy'],
            'f1': val_metrics['f1']
        })
        
        print(f'Epoch {epoch+1}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Metrics:')
        print(f'  Accuracy: {val_metrics["exact_match_accuracy"]:.4f}')
        print(f'  F1: {val_metrics["f1"]:.4f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))

    training_end_time = time.time()
    print(f"Training time: {training_end_time - training_start_time:.2f} seconds")

    # Plot training history
    plot_training_history(metrics_tracker, 1, output_dir)

    # Final evaluation on test set
    print("\nEvaluating on test set...")

    evaluation_start_time = time.time()


    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    test_loss, test_predictions, test_labels_array, overall_metrics, per_category_metrics = evaluate_final(
        model, test_loader, processor.labels, device
    )

    evaluation_end_time = time.time()


    # Create visualizations
    plot_confusion_matrices(test_labels_array, test_predictions, processor.labels, output_dir)
    plot_roc_curves(test_labels_array, test_predictions, processor.labels, output_dir)
    plot_metrics_heatmap(per_category_metrics, processor.labels, output_dir)
    plot_label_distribution(train_labels, test_predictions, processor.labels, output_dir)

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


