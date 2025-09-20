import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pandas as pd
import os
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from multimodal_model import MultimodalViolenceClassifier, MultimodalDataset


class ModelTester:
    """Класс для тестирования и оценки модели"""
    
    def __init__(self, model, device, class_names=['Non-Violent', 'Violent']):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def predict_batch(self, test_loader):
        """Предсказание на тестовой выборке"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                audio_input = batch['audio'].to(self.device)
                visual_input = batch['visual'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Получаем предсказания
                outputs = self.model(audio_input, visual_input)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Получаем эмбеддинги
                embeddings = self.model.get_embeddings(audio_input, visual_input)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_embeddings.extend(embeddings.cpu().numpy())
        
        return (np.array(all_predictions), np.array(all_probabilities), 
                np.array(all_labels), np.array(all_embeddings))
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Вычисление всех метрик"""
        metrics = {}
        
        # Основные метрики
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Метрики для каждого класса
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['auc_roc'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Построение матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Построение ROC кривой"""
        if len(np.unique(y_true)) <= 1:
            print("Недостаточно классов для построения ROC кривой")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return auc_score
    
    def plot_embeddings(self, embeddings, labels, save_path=None):
        """Визуализация эмбеддингов с помощью t-SNE"""
        try:
            from sklearn.manifold import TSNE
            
            # Применяем t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red']
            for i, class_name in enumerate(self.class_names):
                mask = labels == i
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=colors[i], label=class_name, alpha=0.7)
            
            plt.title('t-SNE Visualization of Learned Embeddings')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("scikit-learn не установлен. Пропускаем t-SNE визуализацию.")
    
    def analyze_predictions(self, y_true, y_pred, y_prob, test_loader):
        """Анализ предсказаний модели"""
        # Создаем DataFrame с результатами
        results_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'prob_non_violent': y_prob[:, 0],
            'prob_violent': y_prob[:, 1],
            'correct': y_true == y_pred
        })
        
        # Добавляем названия классов
        results_df['true_class'] = results_df['true_label'].map({0: 'Non-Violent', 1: 'Violent'})
        results_df['predicted_class'] = results_df['predicted_label'].map({0: 'Non-Violent', 1: 'Violent'})
        
        # Статистика по классам
        print("\n=== АНАЛИЗ ПРЕДСКАЗАНИЙ ===")
        print(f"Общая точность: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Количество правильных предсказаний: {sum(results_df['correct'])}")
        print(f"Количество неправильных предсказаний: {sum(~results_df['correct'])}")
        
        # Анализ по классам
        for class_name in self.class_names:
            class_mask = results_df['true_class'] == class_name
            class_correct = results_df[class_mask]['correct'].sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"{class_name}: {class_correct}/{class_total} ({class_acc:.4f})")
        
        # Самые уверенные предсказания
        print("\n=== САМЫЕ УВЕРЕННЫЕ ПРЕДСКАЗАНИЯ ===")
        confident_correct = results_df[results_df['correct']].nlargest(5, 'prob_violent')
        confident_wrong = results_df[~results_df['correct']].nlargest(5, 'prob_violent')
        
        print("Правильные предсказания с высокой уверенностью:")
        for idx, row in confident_correct.iterrows():
            print(f"  {row['true_class']} -> {row['predicted_class']} (conf: {row['prob_violent']:.4f})")
        
        print("\nНеправильные предсказания с высокой уверенностью:")
        for idx, row in confident_wrong.iterrows():
            print(f"  {row['true_class']} -> {row['predicted_class']} (conf: {row['prob_violent']:.4f})")
        
        return results_df
    
    def generate_report(self, y_true, y_pred, y_prob, save_path=None):
        """Генерация подробного отчета"""
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        report = {
            'overall_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'auc_roc': float(metrics['auc_roc'])
            },
            'per_class_metrics': {
                'precision': metrics['precision_per_class'].tolist(),
                'recall': metrics['recall_per_class'].tolist(),
                'f1_score': metrics['f1_per_class'].tolist()
            },
            'class_names': self.class_names,
            'classification_report': classification_report(y_true, y_pred, target_names=self.class_names)
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def test_model(self, test_loader, save_dir="results"):
        """Полное тестирование модели"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Начинаем тестирование модели...")
        print(f"Размер тестовой выборки: {len(test_loader.dataset)}")
        
        # Получаем предсказания
        y_pred, y_prob, y_true, embeddings = self.predict_batch(test_loader)
        
        # Вычисляем метрики
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Выводим основные результаты
        print("\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
        print(f"Точность (Accuracy): {metrics['accuracy']:.4f}")
        print(f"Точность (Precision): {metrics['precision']:.4f}")
        print(f"Полнота (Recall): {metrics['recall']:.4f}")
        print(f"F1-мера: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Детальный отчет по классам
        print("\n=== МЕТРИКИ ПО КЛАССАМ ===")
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics['precision_per_class']):
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
                print(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
                print(f"  F1-score: {metrics['f1_per_class'][i]:.4f}")
            else:
                print(f"{class_name}: Нет данных в тестовой выборке")
        
        # Строим графики
        print("\nСтроим графики...")
        
        # Матрица ошибок
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # ROC кривая
        roc_path = os.path.join(save_dir, 'roc_curve.png')
        self.plot_roc_curve(y_true, y_prob, roc_path)
        
        # Визуализация эмбеддингов
        embeddings_path = os.path.join(save_dir, 'embeddings_tsne.png')
        self.plot_embeddings(embeddings, y_true, embeddings_path)
        
        # Анализ предсказаний
        results_df = self.analyze_predictions(y_true, y_pred, y_prob, test_loader)
        
        # Сохраняем результаты
        results_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
        
        # Генерируем отчет
        report_path = os.path.join(save_dir, 'test_report.json')
        report = self.generate_report(y_true, y_pred, y_prob, report_path)
        
        # Сохраняем эмбеддинги
        np.save(os.path.join(save_dir, 'embeddings.npy'), embeddings)
        
        print(f"\nРезультаты сохранены в директории: {save_dir}")
        
        return metrics, results_df, report


def load_trained_model(model_path, audio_dim, visual_dim, device):
    """Загрузка обученной модели"""
    model = MultimodalViolenceClassifier(audio_dim, visual_dim)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Модель загружена из {model_path}")
    print(f"Эпоха: {checkpoint['epoch']}")
    print(f"Валидационная точность: {checkpoint['val_acc']:.4f}")
    print(f"Валидационная потеря: {checkpoint['val_loss']:.4f}")
    
    return model


def test_multimodal_model(model_path, audio_features_path, visual_features_path, 
                         df_path, scaler_audio_path, scaler_visual_path,
                         save_dir="test_results"):
    """Основная функция для тестирования модели"""
    
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем скейлеры
    with open(scaler_audio_path, 'rb') as f:
        scaler_audio = pickle.load(f)
    
    with open(scaler_visual_path, 'rb') as f:
        scaler_visual = pickle.load(f)
    
    # Создаем тестовый загрузчик данных
    from multimodal_model import create_data_loaders
    _, _, test_loader, _, _ = create_data_loaders(
        audio_features_path, visual_features_path, df_path, batch_size=32
    )
    
    # Получаем размеры входных данных
    sample_batch = next(iter(test_loader))
    audio_dim = sample_batch['audio'].shape[1]
    visual_dim = sample_batch['visual'].shape[1]
    
    # Загружаем модель
    model = load_trained_model(model_path, audio_dim, visual_dim, device)
    
    # Создаем тестер
    tester = ModelTester(model, device)
    
    # Тестируем модель
    metrics, results_df, report = tester.test_model(test_loader, save_dir)
    
    return metrics, results_df, report


if __name__ == "__main__":
    # Тестируем модель
    metrics, results_df, report = test_multimodal_model(
        model_path="models/best_model.pth",
        audio_features_path="audio_features/audio_features.pkl",
        visual_features_path="visual_features/visual_features.pkl",
        df_path="df.csv",
        scaler_audio_path="models/scaler_audio.pkl",
        scaler_visual_path="models/scaler_visual.pkl",
        save_dir="test_results"
    )
