import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.WaterSegmentation import WaterSegmentation
from src.Augmentator import Augmentator

class Unet:
    def __init__(self): 
        self.train_dir = "model_train/"
        self.dataset_dir = 'water_dataset'
        self.segmenter = WaterSegmentation()
        self.augmentator = Augmentator()
        
    def conv_block(self, x, n_filters, kernel_size=3, activation='relu', padding='same', dropout=0.1):
        """
        Сверточный блок для U-Net
        Состоит из двух сверточных слоев с BatchNormalization и
        активацией ReLU.
        """
        x = layers.Conv2D(n_filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(n_filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        
        return x

    def build_unet(self, input_shape=(256, 256, 3), n_filters=64, n_classes=1):
        """
        Построение архитектуры U-Net
        U-Net состоит из двух частей:
        - Encoder (сжатие): уменьшает размер, увеличивает
        количество каналов
        - Decoder (восстановление): увеличивает размер, используя
        skip-connections
        Skip-connections передают информацию от encoder к decoder,
        что позволяет сохранить детали изображения
        """
        inputs = layers.Input(input_shape)

        # Encoder (сжатие)
        c1 = self.conv_block(inputs, n_filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        c2 = self.conv_block(p1, n_filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        c3 = self.conv_block(p2, n_filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        c4 = self.conv_block(p3, n_filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck (дно)
        c5 = self.conv_block(p4, n_filters * 16)
        
        # Decoder (восстановление)
        u6 = layers.Conv2DTranspose(n_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4]) # Skip-connection
        c6 = self.conv_block(u6, n_filters * 8)
        u7 = layers.Conv2DTranspose(n_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3]) # Skip-connection
        c7 = self.conv_block(u7, n_filters * 4)
        u8 = layers.Conv2DTranspose(n_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2]) # Skip-connection
        c8 = self.conv_block(u8, n_filters * 2)
        u9 = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1]) # Skip-connection
        c9 = self.conv_block(u9, n_filters)
        
        # Выходной слой с сигмоидой для бинарной классификации
        outputs = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
        
        return Model(inputs, outputs)

    def dice_loss(self, y_true, y_pred, smooth=1e-6):
        """
        DICE LOSS - функция потерь для сегментации
        Dice Loss = 1 - (2*|A∩B| + smooth) / (|A| + |B| + smooth)
        где A - истинная маска, B - предсказанная маска
        Значение от 0 (идеально) до 1 (плохо)
        """
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        return 1 - score
        
    def tversky_loss(self, y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        TVERSKY LOSS - обобщение Dice Loss с разными весами для
        ошибок
        Параметры:
        - alpha: вес ложноотрицательных ошибок (FN) - пропущенная вода
        - beta: вес ложноположительных ошибок (FP) - ложная тревога
        Для обнаружения разливов важно alpha > beta (пропуск разлива хуже ложной тревоги)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        
        return 1 - tversky
        
    def combined_loss(self, alpha=0.7, beta=0.3):
        """
        Комбинированная функция потерь
        BCE (Binary Cross-Entropy) дает стабильность,
        Tversky дает точность на границах объектов
        """
        def loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            tv = self.tversky_loss(y_true, y_pred, alpha, beta)
            return bce + tv
        return loss

    def demonstrate_unet(self):
        """
        Демонстрация создания U-Net и функций потерь
        """
        print("\n" + "=" * 60)
        print("Этап 4: Реализация U-Net")
        print("=" * 60)
        
        unet = self.build_unet(input_shape=(256, 256, 3), n_filters=16)
        
        print(f"Модель создана! Параметров: {unet.count_params():,}")
        
        unet.summary()
        
        # Тестируем функции потерь на простом примере
        y_true = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
        y_pred = tf.constant([[[[0.9], [0.1]], [[0.2], [0.8]]]])
        
        print(f" Dice Loss: {self.dice_loss(y_true, y_pred).numpy():.4f}")
        print(f" Tversky Loss (α=0.7, β=0.3): {self.tversky_loss(y_true, y_pred).numpy():.4f}")
        
        return unet

    def train_model(self, model, X_train, y_train, X_val, y_val, loss_fn, model_name, epochs=50):
        """
        Обучение модели с заданной функцией потерь
        Используются колбэки:
        - ModelCheckpoint: сохраняет лучшую модель
        - EarlyStopping: останавливает обучение при отсутствии
        прогресса
        - ReduceLROnPlateau: уменьшает learning rate при застое
        """
        print(f"\nОбучение {model_name}...")
        
        # Нормализация данных
        if X_train.max() > 1.0:
            X_train = X_train.astype('float32') / X_train.max()
            X_val = X_val.astype('float32') / X_val.max()
        
        # Метрика IoU для бинарной сегментации
        binary_iou = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss_fn,
            metrics=['accuracy', binary_iou]
        )
        
        callbacks = [
            ModelCheckpoint(f'{self.train_dir}/{model_name}.h5', save_best_only=True, monitor='val_iou', mode='max', verbose=1),
            EarlyStopping(patience=30, restore_best_weights=True, monitor='val_iou', mode='max', verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        augmentor = self.augmentator.get_augmentation(p=0.5)
        print(f"Аугментация train: {X_train.shape[0]} патчей...")
        X_train_aug, y_train_aug = self.augmentator.apply_augmentation(
            X_train, y_train[..., 0] if y_train.ndim == 4 else y_train, augmentor
        )
       
        if y_train_aug.ndim == 3:
            y_train_aug = y_train_aug[..., np.newaxis]

        X_train_aug = X_train_aug.astype(np.float32)
        y_train_aug = y_train_aug.astype(np.float32)

        history = model.fit(X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        batch_size=8, epochs=epochs,
        callbacks=callbacks, verbose=1)
        
        return history

    def plot_training_history(self, history, title):
        """
        Визуализация процесса обучения
        Строит три графика:
        - Loss (функция потерь)
        - Accuracy (точность)
        - IoU (Intersection over Union) - основная метрика
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric in zip(axes, ['loss', 'accuracy', 'iou']):
            ax.plot(history.history[metric], label='Train', linewidth=2)
            ax.plot(history.history[f'val_{metric}'], label='Validation', linewidth=2)
            ax.set_title(f'{title} - {metric}', fontsize=12)
            ax.set_xlabel('Эпоха')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.train_dir}/{title}_training.png', dpi=150, bbox_inches='tight')
        plt.show()

    def evaluate_models(self, models_dict, X_test, y_test):
        """
        Оценка моделей на тестовых данных
        Вычисляет средний IoU, стандартное отклонение, минимум и
        максимум.
        """
        print("\n" + "=" * 60)
        print("Оценка моделей")
        print("=" * 60)

        results = {}
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test, verbose=0)
            y_pred_bin = (y_pred > 0.3).astype(np.float32)
            
            ious = [self.segmenter.calculate_metrics(y_test[i], y_pred_bin[i])['IoU'] for i in range(len(X_test))]

            results[name] = {
                'IoU': np.mean(ious), 
                'IoU_std': np.std(ious),
                'IoU_min': np.min(ious), 
                'IoU_max': np.max(ious)
            }
            
            print(f"\n{name}: IoU = {results[name]['IoU']:.4f} ± {results[name]['IoU_std']:.4f}")

        return results

    def weighted_dice_loss(self, y_true, y_pred, smooth=1e-6, weight=5.0):
        """
        Взвешенный DICE LOSS для корректировки дисбаланса классов
        Увеличивает вес ошибок на воде (которой обычно мало).
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        weights = y_true * weight + (1 - y_true) * 1
        intersection = tf.reduce_sum(weights * y_true * y_pred)
        union = tf.reduce_sum(weights * y_true) + tf.reduce_sum(weights * y_pred)
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice

    def compare_loss_functions(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=30):
        """
        Эксперимент: сравнение функций потерь
        Обучает три модели с разными функциями потерь:
        1 Dice Loss
        2 Tversky Loss
        3 Weighted Dice Loss
        """
        print("\n" + "=" * 60)
        print("Сравнение функций потерь")
        print("=" * 60)

        models, histories = {}, {}

        # 1 Dice Loss
        model_dice = self.build_unet(input_shape=(256, 256, 3), n_filters=32)
        
        history_dice = self.train_model(
            model_dice, 
            X_train, y_train, 
            X_val, y_val,
            self.dice_loss, 
            'unet_dice',
            epochs=epochs
        )

        models['Dice Loss'], histories['Dice Loss'] = model_dice, history_dice
        self.plot_training_history(history_dice, 'Dice_Loss')
        
        # 2 Tversky Loss
        def tversky_loss_07_03(y_true, y_pred):
            return self.tversky_loss(y_true, y_pred, 0.7, 0.3)
    
        model_tversky = self.build_unet(input_shape=(256, 256, 3), n_filters=32)
        history_tversky = self.train_model(
            model_tversky, 
            X_train, y_train, 
            X_val, y_val, 
            tversky_loss_07_03, 
            'unet_tversky'
        )

        models['Tversky Loss'], histories['Tversky Loss'] = model_tversky, history_tversky
        self.plot_training_history(history_tversky, 'Tversky_Loss')
        
        # 3 Weighted Dice Loss
        def weighted_dice_loss_5(y_true, y_pred):
            return self.weighted_dice_loss(y_true, y_pred, weight=5.0)
        
        model_weighted = self.build_unet(input_shape=(256, 256, 3), n_filters=32)
        
        history_weighted = self.train_model(
            model_weighted,
            X_train, y_train, 
            X_val, y_val, 
            weighted_dice_loss_5,
            'unet_weighted_dice_5'
        )

        models['Weighted Dice (w=5)'], histories['Weighted Dice (w=5)'] = model_weighted, history_weighted
        self.plot_training_history(history_weighted, 'Weighted_Dice_w5')

        # 4 Combined Loss
        model_combined = self.build_unet(input_shape=(256, 256, 3), n_filters=32)
        
        history_combined = self.train_model(
            model_combined, 
            X_train, y_train, 
            X_val, y_val,
            self.combined_loss(alpha=0.5, beta=0.5), 
            'combined_loss'
        )

        models['Combined Loss'], histories['Combined Loss'] = model_combined, history_combined
        self.plot_training_history(history_combined, 'Combined_Loss')
        
        # Оценка и визуализация сравнения
        results = self.evaluate_models(models, X_test, y_test)
        self.plot_loss_comparison(results)
        
        return models, histories, results

    def plot_loss_comparison(self, results):
        """
        Визуализация сравнения функций потерь
        Столбчатая диаграмма с погрешностями для трех моделей.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(results.keys())
        ious = [results[n]['IoU'] for n in names]
        errors = [results[n]['IoU_std'] for n in names]
        
        bars = ax.bar(names, ious, yerr=errors, capsize=10, alpha=0.7)
        
        for bar, iou in zip(bars, ious):
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 0.01, 
                f'{iou:.3f}', 
                ha='center', 
                va='bottom', 
                fontsize=11
            )
        
        ax.set_xlabel('Функция потерь')
        ax.set_ylabel('IoU')
        ax.set_title('Сравнение функций потерь')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.train_dir}/loss_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        """
        Визуализация сравнения функций потерь
        Столбчатая диаграмма с погрешностями для трех моделей.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(results.keys())
        ious = [results[n]['IoU'] for n in names]
        errors = [results[n]['IoU_std'] for n in names]
        
        bars = ax.bar(names, ious, yerr=errors, capsize=10, alpha=0.7)
        
        for bar, iou in zip(bars, ious):
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 0.01, 
                f'{iou:.3f}', 
                ha='center', 
                va='bottom', 
                fontsize=11
            )
        
        ax.set_xlabel('Функция потерь')
        ax.set_ylabel('IoU')
        ax.set_title('Сравнение функций потерь')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.train_dir}/loss_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def experiment_tversky_params(self, X_train, y_train, X_val, y_val, param_grid=None, epochs=20, sample_size=100):
        """
        Проверяет разные комбинации alpha и beta:
        - (0.3,0.7): приоритет FP (меньше ложных тревог)
        - (0.5,0.5): равновесный (как Dice)
        - (0.7,0.3): приоритет FN (меньше пропусков) - для экологии
        - (0.8,0.2): сильный приоритет FN
        - (0.9,0.1): экстремальный приоритет FN
        """
        print("\n" + "=" * 60)
        print("Исследование параметров TVERSKY")
        print("=" * 60)
        
        # Используем подвыборку для ускорения
        if len(X_train) > sample_size:
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_s, y_train_s = X_train[idx], y_train[idx]
        else:
            X_train_s, y_train_s = X_train, y_train
            
        if len(X_val) > sample_size // 2:
            idx = np.random.choice(len(X_val), sample_size // 2, replace=False)
            X_val_s, y_val_s = X_val[idx], y_val[idx]
        else:
            X_val_s, y_val_s = X_val, y_val
            
        param_grid = param_grid or [
            (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
        ]
        
        results = []
        
        for alpha, beta in param_grid:
            print(f"\nТестирование α={alpha}, β={beta}...")
            model = self.build_unet(input_shape=(256, 256, 3), n_filters=32)
            
            def tversky_loss_current(y_true, y_pred):
                return self.tversky_loss(y_true, y_pred, alpha, beta)
                
            model.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss=tversky_loss_current)
            model.fit(X_train_s, y_train_s,
                        validation_data=(X_val_s, y_val_s),
                        batch_size=8, epochs=epochs, verbose=0)
                        
            y_pred = model.predict(X_val_s, verbose=0)
            y_pred_bin = (y_pred > 0.5).astype(np.float32)
            
            ious, recalls, precisions = [], [], []
            
            for i in range(len(X_val_s)):
                metrics = self.segmenter.calculate_metrics(y_val_s[i], y_pred_bin[i])
                ious.append(metrics['IoU'])
                
                true_bin = y_val_s[i] > 0.5
                pred_bin = y_pred_bin[i] > 0.3
                
                tp = np.sum(true_bin & pred_bin)
                fp = np.sum(~true_bin & pred_bin)
                fn = np.sum(true_bin & ~pred_bin)
                
                recalls.append(tp / (tp + fn + 1e-10))
                precisions.append(tp / (tp + fp + 1e-10))
                
            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)
            
            results.append({
                'alpha': alpha, 'beta': beta,
                'IoU': np.mean(ious), 'Recall': mean_recall,
                'Precision': mean_precision,
                'F1': 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-10)
            })
            
            print(f" IoU: {results[-1]['IoU']:.4f}, F1: {results[-1]['F1']:.4f}")
            
        results_df = pd.DataFrame(results)
        self.plot_tversky_experiment(results_df)
        results_df.to_csv(f'{self.dataset_dir}/tversky_experiment.csv', index=False)
        
        return results_df
    
    def plot_tversky_experiment(self, results_df):
        """
        Визуализация результатов эксперимента С TVERSKY LOSS
        Два графика:
        - Сравнение IoU, Recall, Precision для разных параметров
        - F1-score для выбора лучших параметров
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = [f"α={r:.1f}\nβ={b:.1f}" for r, b in zip(results_df['alpha'], results_df['beta'])]
        x_pos = np.arange(len(x))
        width = 0.25
        
        # График 1: Метрики
        ax1.bar(x_pos - width, results_df['IoU'], width, label='IoU', alpha=0.8, color='green')
        ax1.bar(x_pos, results_df['Recall'], width, label='Recall', alpha=0.8, color='blue')
        ax1.bar(x_pos + width, results_df['Precision'], width, label='Precision', alpha=0.8, color='red')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: F1-score
        best_idx = results_df['F1'].argmax()
        ax2.bar(x_pos, results_df['F1'], width, alpha=0.8, color='orange')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.axhline(y=results_df['F1'].max(), color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/tversky_experiment.png', dpi=150, bbox_inches='tight')
        plt.show()