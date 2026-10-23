from src.WaterQualityPipelineExtended import WaterQualityPipelineExtended
from src.WaterSegmentation import WaterSegmentation
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import warnings
import random
from PIL import Image
from skimage.filters import threshold_otsu, threshold_local
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage import exposure
import albumentations as A
import cv2

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

    def create_dataset_from_pract2(self, raster_path='data/degtyarsk_water.tif', output_dir='water_dataset', patch_size=256, stride=32, save_4band=True):
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
                
                # Создаем эталонную маску по NDWI (индекс воды)
                # NDWI = (Green - NIR) / (Green + NIR) дает значения >0.15 для воды
                ndwi = (green - nir) / (green + nir + 1e-10)
                water_mask = (ndwi > 0.15).astype(np.uint8)

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
        
        files = sorted(set(os.listdir(image_dir)) &
        set(os.listdir(mask_dir)))
        n_samples = min(n_samples, len(files))
        selected = random.sample(files, n_samples)
        selected.sort()
        
        # Загружаем данные
        images_rgb, true_masks, images_4band = [], [], []
        for fname in selected:
            images_rgb.append(plt.imread(f'{image_dir}/{fname}'))
            mask = plt.imread(f'{mask_dir}/{fname}')
            true_masks.append(mask[:, :, 0] if mask.ndim == 3 else mask)
            npy_path = f'{npy_dir}/{fname[:-4]}.npy'

            if os.path.exists(npy_path):
                images_4band.append(np.load(npy_path))
            else:
                images_4band.append(np.zeros((256, 256, 4), dtype=np.float32))

        # Применяем все методы
        method_results = {
            'Глобальный (0.15)': [self.segmenter.global_threshold(img4) for img4 in images_4band],
            'Оцу': [self.segmenter.otsu_threshold(img4) for img4 in images_4band],
            'Адаптивный': [self.segmenter.adaptive_threshold(img4) for img4 in images_4band],
            'NDWI (0.15)': [self.segmenter.ndwi_threshold(img4) for img4 in images_4band]
        }
        
        # Добавляем постобработку
        postprocessed = {}
        for method_name, preds in method_results.items():
            post_preds = [self.segmenter.postprocess_mask(pred) for pred in preds]
            postprocessed[f'{method_name}+Пост'] = post_preds

        method_results.update(postprocessed)
        
        # Визуализируем результаты
        self._show_results(images_rgb, true_masks, method_results)
        
        # Выводим метрики
        print("\n" + "=" * 60)
        print("Итоговые метрики (средний IoU по патчам)")
        print("=" * 60)
        
        for method_name in method_results.keys():
            ious = [self.segmenter.calculate_metrics(true_masks[i], method_results[method_name][i])['IoU'] for i in range(len(images_rgb))]
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

    def get_augmentation(self, p=0.5):
        """
        Аугментация позволяет искусственно увеличить размер
        датасета за счет случайных преобразований изображений.
        """
        return A.Compose([
            # Геометрические преобразования
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.3),
            
            # Цветовые преобразования
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
            
            # Шумы и размытие
            A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
            
            # Погодные эффекты (для спутниковых снимков)
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.2),
            A.RandomRain(slant_range=(-10, 10), drop_width=1, p=0.1),
            A.RandomSnow(snow_point_range=(0.1, 0.3), p=0.1),
            
            # Вырезание (учат модель не полагаться на отдельные пиксели)
            A.CoarseDropout(
                num_holes_range=(4, 8), 
                hole_height_range=(16, 32), 
                hole_width_range=(16, 32), 
                fill=0, 
                p=0.2
            ),
        ], p=p)

    def apply_augmentation(self, images, masks, augmentor=None):
        """
        Применение аугментации к батчу изображений
        Преобразует изображения и соответствующие маски, сохраняя
        их соответствие.
        """
        if augmentor is None:
            augmentor = self.get_augmentation()
        
        aug_images, aug_masks = [], []
        for i in range(len(images)):
            # Приводим к uint8
            img_uint8 = (images[i] * 255).astype(np.uint8) if images[i].max() <= 1 else images[i].astype(np.uint8)
            mask_uint8 = (masks[i] * 255).astype(np.uint8)
            augmented = augmentor(image=img_uint8, mask=mask_uint8)

            # Возвращаем в float32 [0,1] для нейросети
            aug_images.append(augmented['image'].astype(np.float32) / 255.0)
            aug_masks.append(augmented['mask'].astype(np.float32) / 255.0)
        
        return np.array(aug_images), np.array(aug_masks)

    def visualize_augmentation(self, n_samples=4):
        """
        Демонстрация аугментации на примерах
        Показывает оригиналы и аугментированные версии изображений
        и масок.
        """
        print("\n" + "=" * 60)
        print("Демонстрация аугментации")
        print("=" * 60)
        
        if not hasattr(self, 'dataset') or 'X_train' not in self.dataset:
            print("Сначала выполните load_dataset()")
            return
        
        sample_images = self.dataset['X_train'][:n_samples]
        sample_masks = self.dataset['y_train'][:n_samples]
        aug_images, aug_masks = self.apply_augmentation(sample_images, sample_masks)
        fig, axes = plt.subplots(2, n_samples * 2, figsize=(4 * n_samples * 2, 8))
        
        for i in range(n_samples):
            # Оригиналы
            axes[0, i * 2].imshow(sample_images[i])
            axes[0, i * 2].set_title(f'Оригинал {i + 1}')
            axes[0, i * 2].axis('off')
            axes[0, i * 2 + 1].imshow(sample_masks[i], cmap='Blues', vmin=0, vmax=1)
            axes[0, i * 2 + 1].set_title(f'Маска {i + 1}')
            axes[0, i * 2 + 1].axis('off')
            
            # Аугментированные
            axes[1, i * 2].imshow(aug_images[i])
            axes[1, i * 2].set_title(f'Аугмент. {i + 1}')
            axes[1, i * 2].axis('off')
            axes[1, i * 2 + 1].imshow(aug_masks[i], cmap='Blues', vmin=0, vmax=1)
            axes[1, i * 2 + 1].set_title(f'Маска аугм.')
            axes[1, i * 2 + 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.dataset_dir}/augmentation_demo.png', dpi=150, bbox_inches='tight')
        plt.show()