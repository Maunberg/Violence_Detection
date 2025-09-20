import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os
import pickle
from tqdm import tqdm
import json
from datetime import datetime

from multimodal_model import MultimodalViolenceClassifier, create_data_loaders


class EarlyStopping:
    """Ранняя остановка для предотвращения переобучения"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """Класс для обучения мультимодальной модели"""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Ранняя остановка
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    def train_epoch(self, train_loader):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            audio_input = batch['audio'].to(self.device)
            visual_input = batch['visual'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(audio_input, visual_input)
            loss = self.criterion(outputs, labels)
            
            # Обратный проход
            loss.backward()
            
            # Градиентное обрезание
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Обновляем прогресс бар
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Валидация на одной эпохе"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                audio_input = batch['audio'].to(self.device)
                visual_input = batch['visual'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Прямой проход
                outputs = self.model(audio_input, visual_input)
                loss = self.criterion(outputs, labels)
                
                # Статистика
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Сохраняем предсказания для метрик
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Дополнительные метрики
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=100, save_dir="models"):
        """Полное обучение модели"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Начинаем обучение на {num_epochs} эпох...")
        print(f"Устройство: {self.device}")
        print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
        print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_acc, val_precision, val_recall, val_f1, val_pred, val_labels = self.validate_epoch(val_loader)
            
            # Обновляем планировщик
            self.scheduler.step(val_loss)
            
            # Сохраняем историю
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Выводим статистику
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc
                }, os.path.join(save_dir, 'best_model.pth'))
                print("✓ Сохранена лучшая модель")
            
            # Ранняя остановка
            if self.early_stopping(val_loss, self.model):
                print(f"Ранняя остановка на эпохе {epoch + 1}")
                break
        
        # Сохраняем финальную модель
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc
        }, os.path.join(save_dir, 'final_model.pth'))
        
        # Сохраняем историю обучения
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nОбучение завершено!")
        print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
        print(f"Лучшая валидационная потеря: {best_val_loss:.4f}")
        
        return history
    
    def plot_training_history(self, save_dir="models"):
        """Построение графиков истории обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График точности
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()


def train_multimodal_model(audio_features_path, visual_features_path, df_path, 
                          num_epochs=100, batch_size=32, learning_rate=0.001,
                          save_dir="models"):
    """Основная функция для обучения модели"""
    
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем загрузчики данных
    print("Загружаем данные...")
    train_loader, val_loader, test_loader, scaler_audio, scaler_visual = create_data_loaders(
        audio_features_path, visual_features_path, df_path, batch_size=batch_size
    )
    
    # Получаем размеры входных данных
    sample_batch = next(iter(train_loader))
    audio_dim = sample_batch['audio'].shape[1]
    visual_dim = sample_batch['visual'].shape[1]
    
    print(f"Размер аудио фичей: {audio_dim}")
    print(f"Размер визуальных фичей: {visual_dim}")
    
    # Создаем модель
    model = MultimodalViolenceClassifier(audio_dim, visual_dim)
    print(f"Модель создана. Параметров: {sum(p.numel() for p in model.parameters())}")
    
    # Создаем тренер
    trainer = ModelTrainer(model, device, learning_rate=learning_rate)
    
    # Обучаем модель
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs, save_dir=save_dir)
    
    # Строим графики
    trainer.plot_training_history(save_dir)
    
    # Сохраняем скейлеры
    with open(os.path.join(save_dir, 'scaler_audio.pkl'), 'wb') as f:
        pickle.dump(scaler_audio, f)
    
    with open(os.path.join(save_dir, 'scaler_visual.pkl'), 'wb') as f:
        pickle.dump(scaler_visual, f)
    
    return model, history, test_loader


if __name__ == "__main__":
    # Обучаем модель
    model, history, test_loader = train_multimodal_model(
        audio_features_path="audio_features/audio_features.pkl",
        visual_features_path="visual_features/visual_features.pkl",
        df_path="df.csv",
        num_epochs=50,
        batch_size=16,
        learning_rate=0.001,
        save_dir="models"
    )
