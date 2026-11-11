from src.WaterQualityPipelineExtended import WaterQualityPipelineExtended
from src.WaterSegmentation import WaterSegmentation
from shapely.geometry import Point, box
from rasterio.mask import mask
from rasterio.plot import show
from PIL import Image
from skimage.filters import threshold_otsu, threshold_local
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage import exposure
import os
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import tensorflow as tf
import geopandas as gpd
import pandas as pd
import rasterio
import warnings
import random
import cv2
import json

class WaterQualityPipelineSegExt(WaterQualityPipelineExtended): 
    def __init__(self, name="Сегментация"): 
        """Конструктор класса""" 
        super().__init__(name) 
        self.name = name 
        self.dataset_dir = 'water_dataset'      # Папка с датасетом 
        self.train_dir = 'model_train'          # Папка для сохранения моделей
        self.segmenter = WaterSegmentation()    # Объект для пороговых методов

    def load_dataset(self, data_dir='water_dataset', img_size=(256, 256), test_size=0.2, val_size=0.2):
        """
        Разделяет данные на три части:
        - Train (обучение) - 80% (при test_size=0.2)
        - Val (валидация) - 4% (20% от 20%)
        - Test (тест) - 16% (80% от 20%)
        """

        print("\n" + "=" * 60)
        print("Загрузка и разделение датасета")
        print("=" * 60)

        image_dir = f'{data_dir}/images'
        mask_dir = f'{data_dir}/masks'

        if not (os.path.exists(image_dir) and os.path.exists(mask_dir)):
            print("Сначала создайте датасет")
            return None

        common_files = sorted(set(os.listdir(image_dir)) & set(os.listdir(mask_dir)))
        print(f"Найдено {len(common_files)} пар изображение-маска")
        
        images, masks = [], []
        for img_file, mask_file in zip(sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir))):
            if img_file not in common_files or mask_file not in common_files:
                continue

            # Загрузка и нормализация изображения
            img = plt.imread(f'{image_dir}/{img_file}')
            if img.max() > 1.0:
                if img.max() <= 255:
                    img = img.astype(np.float32) / 255
                else:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            # Ресайз при необходимости
            if img.shape[:2] != img_size:
                img = resize(img, img_size, preserve_range=True, anti_aliasing=True)
            
            # Загрузка маски
            mask = plt.imread(f'{mask_dir}/{mask_file}')
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.shape[:2] != img_size:
                mask = resize(mask, img_size, preserve_range=True, anti_aliasing=False)
                mask = (mask > 0.5).astype(np.float32)
            
            images.append(img.astype(np.float32))
            masks.append(mask)
            
        X, y = np.array(images), np.array(masks)
        print(f"Размерность данных: X {X.shape}, y {y.shape}")
        
        # Разделение на train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_size, random_state=42)
        
        print(f"\nРазделение: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")
        
        self.dataset = {
            'X_train': X_train, 'y_train': y_train, 
            'X_val': X_val, 'y_val': y_val, 
            'X_test': X_test, 'y_test': y_test
            }

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def load_reference_mask(self, src, reference_path, water_classes=(0,)):
        """
        Маска воды по карте классов Dynamic World (0 - вода).
        Маска должна быть на той же сетке, что и снимок
        """
        with rasterio.open(reference_path) as ref:
            if ref.crs != src.crs or ref.transform != src.transform or ref.shape != src.shape:
                raise ValueError("Сетка маски не совпадает со снимком")
            classes = ref.read(1)
        return np.isin(classes, water_classes).astype(np.uint8)

    def create_dataset_from_pract2(self, raster_path='data/degtyarsk_water.tif', dynamic_world_path='data/dynamic_world_label.tif', 
                                   output_dir='water_dataset', patch_size=256, stride=32, save_4band=True):
        """
        Использует спутниковый снимок из Практической работы №2 и создает набор патчей (изображений) с соответствующими масками воды.
        Параметры:
        - raster_path: путь к GeoTIFF снимку
        - output_dir: папка для сохранения датасета
        - patch_size: размер патча (обычно 256x256)
        - stride: шаг нарезки (меньше шаг = больше патчей)
        - save_4band: сохранять ли 4-канальные данные (RGB + NIR)
        """
        print("\n" + "=" * 60)
        print("Задача 3.4: Создание датасет для сегментации")
        print("=" * 60)
        self.dataset_dir = output_dir
        
        # Создаем необходимые папки
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        os.makedirs(f'{output_dir}/masks', exist_ok=True)
        
        if save_4band:
            os.makedirs(f'{output_dir}/images_4band', exist_ok=True)
        try:
            with rasterio.open(raster_path) as src:
                # Читаем каналы спутникового снимка
                red = src.read(1).astype(np.float32) # B4 - красный
                green = src.read(2).astype(np.float32) # B3 - зеленый
                blue = src.read(3).astype(np.float32) # B2 - синий
                nir = src.read(4).astype(np.float32) # B8 - ближний инфракрасный
                
                water_mask = self.load_reference_mask(src, dynamic_world_path)

                # Функция нормализации каналов для визуализации (отсекаем 2% и 98% процентили)
                def norm_band(band):
                    p2, p98 = np.percentile(band, (2, 98))
                    return np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1) * 255
                
                # Создаем RGB изображение для визуализации
                rgb = np.stack([norm_band(red), norm_band(green), norm_band(blue)], axis=-1).astype(np.uint8)

                # Нарезаем снимок на патчи
                h, w = rgb.shape[:2]
                patch_idx = 0
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        mask_patch = water_mask[y:y + patch_size, x:x + patch_size]
                        
                        # Сохраняем только патчи, где есть вода (больше 1% пикселей)
                        if np.sum(mask_patch) > patch_size ** 2 * 0.01:
                            
                            # Сохраняем RGB патч
                            Image.fromarray(rgb[y:y + patch_size, x:x + patch_size]).save(f'{output_dir}/images/patch_{patch_idx:04d}.png')
                            
                            # Сохраняем 4-канальные данные для NDWI (R,G,B,NIR)
                            if save_4band:
                                bands = np.stack([
                                    red[y:y + patch_size, x:x + patch_size],
                                    green[y:y + patch_size, x:x + patch_size],
                                    blue[y:y + patch_size, x:x + patch_size],
                                    nir[y:y + patch_size, x:x + patch_size]
                                ])

                                np.save(f'{output_dir}/images_4band/patch_{patch_idx:04d}.npy', bands)

                            # Сохраняем маску воды
                            Image.fromarray((mask_patch * 255).astype(np.uint8)).save(f'{output_dir}/masks/patch_{patch_idx:04d}.png')
                            patch_idx += 1
                
                print(f"Создано {patch_idx} патчей")

                print(f" - images/: RGB изображения для визуализации")
                print(f" - masks/: бинарные маски воды (1 - вода, 0 - суша)")
                if save_4band:
                    print(f" - images_4band/: 4-канальные данные для NDWI")
                    return patch_idx
        
        except Exception as e:
            print(f"Ошибка при создании датасета: {e}")
        
        return 0

    def test_methods_on_patches(self, n_samples=3):
        """
        в class
        WaterQualityPipelineSegExt(WaterQualityPipeline):
        Тестирование методов сегментации на патчах
        Сравнивает различные пороговые методы сегментации:
        - Глобальный порог по NDWI
        - Метод Оцу
        - Адаптивный порог
        - NDWI с порогом 0.15
        """
        print("\n" + "=" * 60)
        print("Задача 1.2: Тестирование методов сегментации")
        print("=" * 60)

        image_dir = f'{self.dataset_dir}/images'
        mask_dir = f'{self.dataset_dir}/masks'
        npy_dir = f'{self.dataset_dir}/images_4band'
        
        if not all(os.path.exists(d) for d in [image_dir, mask_dir, npy_dir]):
            print("Сначала создайте датасет с 4-канальными данными")
            return
        
        files = sorted(set(os.listdir(image_dir)) & set(os.listdir(mask_dir)))
        n_samples = min(n_samples, len(files))
        selected = sorted(random.sample(files, n_samples))
        
        # Загружаем данные
        true_masks, images_4band = [], []
        for fname in files:
            mask = plt.imread(f'{mask_dir}/{fname}')
            true_masks.append(mask[:, :, 0] if mask.ndim == 3 else mask)
            npy_path = f'{npy_dir}/{fname[:-4]}.npy'
            if os.path.exists(npy_path):
                images_4band.append(np.load(npy_path))
            else:
                images_4band.append(np.zeros((256, 256, 4), dtype=np.float32))

        # Применяем все методы
        method_results = {
            'Глобальный (0.5)': [self.segmenter.global_threshold(img4) for img4 in images_4band],
            'Оцу': [self.segmenter.otsu_threshold(img4) for img4 in images_4band],
            'Адаптивный': [self.segmenter.adaptive_threshold(img4) for img4 in images_4band],
            'NDWI (0.15)': [self.segmenter.ndwi_threshold(img4) for img4 in images_4band]
        }
        
        # Добавляем постобработку
        postprocessed = {}
        for method_name, preds in method_results.items():
            postprocessed[f'{method_name}+Пост'] = [self.segmenter.postprocess_mask(pred) for pred in preds]
        
        method_results.update(postprocessed)
        
        # Визуализируем результаты
        sel_idx = [files.index(f) for f in selected]
        images_rgb = [plt.imread(f'{image_dir}/{fname}') for fname in selected]
        true_masks_vis = [true_masks[i] for i in sel_idx]
        method_results_vis = {name: [preds[i] for i in sel_idx] for name, preds in method_results.items()}

        self._show_results(images_rgb, true_masks_vis, method_results_vis)
        
        # Выводим метрики
        print("\n" + "=" * 60)
        print(f"Итоговые метрики (средний IoU по всем {len(files)} патчам)")
        print("=" * 60)
        
        for method_name in method_results.keys():
            ious = [self.segmenter.calculate_metrics(true_masks[i], method_results[method_name][i])['IoU']
                    for i in range(len(files))]
            print(f"{method_name:25}: {np.mean(ious):.3f}")

    
    def _show_results(self, images, true_masks, methods):
        """
        в класс WaterQualityPipelineSegExt
        Вспомогательный метод для визуализации результатов
        Создает сетку изображений: оригинал, идеальная маска и
        результаты всех методов.
        """
        n_samples = len(images)
        n_methods = len(methods)
        
        fig, axes = plt.subplots(n_samples, n_methods + 2, figsize=(4 * (n_methods + 2), 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        print("\nIoU по методам (для первого патча):")
        print("-" * 50)
        
        for i in range(n_samples):
            # Оригинал
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('RGB', fontsize=12)
            axes[i, 0].axis('off')
            
            # Идеальная маска (Ground Truth)
            axes[i, 1].imshow(true_masks[i], cmap='Blues', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth', fontsize=12)
            axes[i, 1].axis('off')
            
            # Результаты всех методов
            for j, (method_name, preds) in enumerate(methods.items()):
                col = j + 2
                pred = preds[i]
                
                # Приводим к правильной размерности
                if len(pred.shape) == 3 and pred.shape[0] == 4:
                    pred = np.mean(pred, axis=0)
                elif len(pred.shape) == 3:
                    pred = np.mean(pred, axis=-1)
                
                if len(pred.shape) == 2 and pred.shape == true_masks[i].shape:
                    iou = self.segmenter.calculate_metrics(true_masks[i], pred)['IoU']
                    axes[i, col].imshow(pred, cmap='Blues', vmin=0, vmax=1)
                    axes[i, col].set_title(f'{method_name}\nIoU:{iou:.3f}', fontsize=10)
                    
                    if i == 0:
                        print(f"{method_name:25}: {iou:.3f}")
                
                else:
                    axes[i, col].imshow(np.zeros_like(true_masks[i]), cmap='Blues')
                    axes[i, col].set_title(f'{method_name}\nОшибка', fontsize=10)
                
                axes[i, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/segmentation_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    def find_optimal_threshold(self, model, X_val, y_val, model_name="", thresholds=np.arange(0.3, 0.9, 0.05)):
        """
        Поиск оптимального порога бинаризации для модели
        Перебираем пороги от 0.3 до 0.85 и выбираем тот,
        при котором IoU на валидационной выборке максимален.
        """
        print(f"\nПоиск оптимального порога для {model_name}...")
        best_threshold, best_iou = 0.5, 0
        threshold_results = []

        n_samples = min(30, len(X_val))
        X_sample, y_sample = X_val[:n_samples], y_val[:n_samples]

        for threshold in thresholds:
            ious = []
            for i in range(n_samples):
                pred = model.predict(X_sample[i:i + 1], verbose=0)[0, :, :, 0]
                pred_bin = self.segmenter.postprocess_mask((pred > threshold).astype(np.float32))
                iou = self.segmenter.calculate_metrics(y_sample[i], pred_bin)['IoU']
                ious.append(iou)

            mean_iou = np.mean(ious)
            threshold_results.append((threshold, mean_iou))

            if mean_iou > best_iou:
                best_iou, best_threshold = mean_iou, threshold
                print(f"Порог {threshold:.2f}: IoU = {mean_iou:.4f} (НОВЫЙ ЛУЧШИЙ)")
            else:
                print(f"Порог {threshold:.2f}: IoU = {mean_iou:.4f}")

        self.plot_threshold_analysis(threshold_results, model_name)
        return best_threshold, best_iou

    def plot_threshold_analysis(self, threshold_results, model_name):
        """
        ВИЗУАЛИЗАЦИЯ ЗАВИСИМОСТИ IoU ОТ ПОРОГА
        Показываем, как меняется качество сегментации при изменении
        порога, и отмечаем оптимальное значение.
        """
        thresholds, ious = zip(*threshold_results)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, ious, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Порог бинаризации')
        plt.ylabel('IoU')
        plt.title(f'Зависимость качества от порога\n{model_name}')
        plt.grid(True, alpha=0.3)
    
        best_idx = np.argmax(ious)
        best_threshold, best_iou = thresholds[best_idx], ious[best_idx]
    
        plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7)
        plt.scatter([best_threshold], [best_iou], color='red', s=200, zorder=5)
    
        plt.annotate(
            f'Оптимум: {best_threshold:.2f}\nIoU={best_iou:.4f}',
            xy=(best_threshold, best_iou), xytext=(10, -35), textcoords='offset points',
            fontsize=10, color='red'
        )
    
        plt.tight_layout()
        plt.savefig(f'{self.train_dir}/threshold_analysis_{model_name.replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()


    def visualize_predictions(self, models_dict, X_test, y_test, n_samples=4, save=True):
        """
        Визуализация предсказаний моделей с адаптивными порогами
        Показывает для каждого тестового примера:
        - оригинал
        - идеальную маску
        - предсказания всех моделей
        - карту ошибок лучшей модели
        """
        print("\n" + "=" * 60)
        print("Визуализация предсказаний")
        print("=" * 60)
        
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        n_models = len(models_dict)
        
        fig, axes = plt.subplots(len(indices), n_models + 3, figsize=(4 * (n_models + 3), 4 * len(indices)))
        if len(indices) == 1:
            axes = axes.reshape(1, -1)
            
        # Эмпирически подобранные пороги для каждой модели
        optimal_thresholds = {
            'Dice Loss': 0.65, 
            'Tversky Loss': 0.5, 
            'Weighted Dice (w=5)': 0.45
        }
        
        for row, idx in enumerate(indices):
            img, true_mask = X_test[idx], y_test[idx].squeeze()
            col = 0
            
            # Оригинал
            axes[row, col].imshow(img)
            axes[row, col].set_title('Снимок')
            axes[row, col].axis('off')
            col += 1
            
            # Идеальная маска
            axes[row, col].imshow(true_mask, cmap='Blues', vmin=0, vmax=1)
            axes[row, col].set_title('Идеальная маска')
            axes[row, col].axis('off')
            col += 1
            
            ious = []
            for name, model in models_dict.items():
                pred_raw = model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(name, 0.5)
                
                # Для Dice Loss используем Otsu если предсказания слишком высокие
                if name == 'Dice Loss' and pred_raw.mean() > 0.7:
                    try:
                        pred_norm = ((pred_raw - pred_raw.min()) / (pred_raw.max() - pred_raw.min() + 1e-8) * 255).astype(np.uint8)
                        threshold = threshold_otsu(pred_norm) / 255
                    except:
                        threshold = 0.7
                        
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))
                iou = self.segmenter.calculate_metrics(true_mask, pred_bin)['IoU']
                ious.append((name, iou))
                
                axes[row, col].imshow(pred_bin, cmap='Blues', vmin=0, vmax=1)
                axes[row, col].set_title(f'{name}\nпорог={threshold:.2f}\nIoU={iou:.3f}', fontsize=9)
                axes[row, col].axis('off')
                col += 1
                
            # Карта ошибок лучшей модели
            if ious:
                best_name, best_iou = max(ious, key=lambda x: x[1])
                best_model = models_dict[best_name]
                pred_raw = best_model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(best_name, 0.5)
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))
                
                error_map = np.zeros((*pred_bin.shape, 3))
                tp_mask = (true_mask > 0.5) & (pred_bin > 0.5)  # True Positive (зеленый)
                fp_mask = (true_mask <= 0.5) & (pred_bin > 0.5) # False Positive (красный)
                fn_mask = (true_mask > 0.5) & (pred_bin <= 0.5) # False Negative (желтый)
                
                error_map[tp_mask] = [0, 1, 0]
                error_map[fp_mask] = [1, 0, 0]
                error_map[fn_mask] = [1, 1, 0]
                
                axes[row, -1].imshow(error_map)
                axes[row, -1].set_title(f'Ошибки\n{best_name}: {best_iou:.3f}', fontsize=10)
                axes[row, -1].axis('off')
                
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.dataset_dir}/test_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_predictions_with_thresholds(self, models_dict, X_test, y_test, optimal_thresholds=None, n_samples=4, save=True):
        """
        Визуализация предсказаний с оптимальными порогами
        Использует найденные ранее оптимальные пороги для каждой модели.
        """
        print("\n" + "=" * 60)
        print("Визуализация - оптимальные пороги")
        print("=" * 60)
        
        if optimal_thresholds is None:
            optimal_thresholds = {name: 0.5 for name in models_dict.keys()}
            
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        n_models = len(models_dict)
        
        fig, axes = plt.subplots(len(indices), n_models + 3, figsize=(4 * (n_models + 3), 4 * len(indices)))
        if len(indices) == 1:
            axes = axes.reshape(1, -1)
            
        for row, idx in enumerate(indices):
            img, true_mask = X_test[idx], y_test[idx].squeeze()
            col = 0
            
            axes[row, col].imshow(img)
            axes[row, col].set_title('Снимок')
            axes[row, col].axis('off')
            col += 1
            
            axes[row, col].imshow(true_mask, cmap='Blues', vmin=0, vmax=1)
            axes[row, col].set_title('Идеальная маска')
            axes[row, col].axis('off')
            col += 1
            
            ious = []
            for name, model in models_dict.items():
                pred_raw = model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(name, 0.5)
                
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))
                iou = self.segmenter.calculate_metrics(true_mask, pred_bin)['IoU']
                ious.append((name, iou))
                
                axes[row, col].imshow(pred_bin, cmap='Blues', vmin=0, vmax=1)
                axes[row, col].set_title(f'{name}\nпорог={threshold:.2f}\nIoU={iou:.3f}', fontsize=9)
                axes[row, col].axis('off')
                col += 1
                
            # Карта ошибок лучшей модели
            if ious:
                best_name, best_iou = max(ious, key=lambda x: x[1])
                best_model = models_dict[best_name]
                pred_raw = best_model.predict(img[np.newaxis], verbose=0)[0, :, :, 0]
                threshold = optimal_thresholds.get(best_name, 0.5)
                pred_bin = self.segmenter.postprocess_mask((pred_raw > threshold).astype(np.float32))
                
                error_map = np.zeros((*pred_bin.shape, 3))
                tp_mask = (true_mask > 0.5) & (pred_bin > 0.5)
                fp_mask = (true_mask <= 0.5) & (pred_bin > 0.5)
                fn_mask = (true_mask > 0.5) & (pred_bin <= 0.5)
                
                error_map[tp_mask] = [0, 1, 0]
                error_map[fp_mask] = [1, 0, 0]
                error_map[fn_mask] = [1, 1, 0]
                
                axes[row, -1].imshow(error_map)
                axes[row, -1].set_title(f'Ошибки\n{best_name}: {best_iou:.3f}', fontsize=10)
                axes[row, -1].axis('off')
                
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.dataset_dir}/test_predictions_optimal.png', dpi=150, bbox_inches='tight')
        plt.show()