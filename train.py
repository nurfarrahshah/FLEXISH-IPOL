import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import create_model
import time


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(outputs, 1)
        running_corrects += torch.sum(preds == y.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    val_corrects = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            preds = torch.argmax(outputs, 1)
            
            val_loss += loss.item() * x.size(0)
            val_corrects += torch.sum(preds == y.data)
            val_total += y.size(0)
    
    val_loss = val_loss / val_total if val_total > 0 else 0
    val_acc = val_corrects.double() / val_total if val_total > 0 else 0
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Track parameters for Flexish
        if hasattr(model, 'track_parameters'):
            model.track_parameters(epoch + 1)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Check if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'Time: {epoch_time:.2f}s')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history
