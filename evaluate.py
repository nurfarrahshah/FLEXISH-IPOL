import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def evaluate_model(model, data_loader, device=None):
    """Evaluate model and return comprehensive metrics"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, 1)
            preds = torch.argmax(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    num_classes = all_probs.shape[1]
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['precision'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # AUC calculation
    if num_classes == 2:
        metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        try:
            y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
            metrics['auc'] = roc_auc_score(y_true_bin, all_probs, average='weighted', multi_class='ovr')
        except:
            metrics['auc'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    
    return metrics, all_preds, all_probs


def analyze_activations(model, data_loader, device=None):
    """Analyze activation function behavior"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Hook to capture activations
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())
    
    # Register hooks to Flexish layers
    from flexish import Flexish
    hooks = []
    for module in model.modules():
        if isinstance(module, Flexish):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _ = model(x)
            break  # Just one batch
    
    # Analyze activations
    if activations:
        total_neurons = sum(act.numel() for act in activations)
        dead_neurons = sum((act == 0).sum().item() for act in activations)
        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0
    else:
        dead_ratio = 0
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return dead_ratio
