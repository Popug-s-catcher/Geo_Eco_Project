import json
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.WaterQualityPipelineSegExt import WaterQualityPipelineSegExt
from src.Unet import Unet


class SegmentationPipeline(WaterQualityPipelineSegExt):
    """
    Расширение пайплайна методами сегментации для работы с новыми снимками.
    Позволяет загрузить обученную модель, применить ее к любому спутниковому снимку
    и сохранить результат в виде GeoTIFF и GeoJSON.
    """
    
    def __init__(self, name="Интеграция сегментации"):
        super().__init__(name)
        self.segmentation_model = None
        self.segmentation_masks = None
        self.best_threshold = None # добавили поле для порога
        self.unet = Unet()

    def load_segmentation_model(self, model_path='model_train/unet_tversky.h5'):
        """
        Загрузка обученной модели сегментации.
        Учитывает пользовательские функции потерь, определенные в классе.
        """
        # Словарь с пользовательскими объектами
        custom_objs = {
            'dice_loss': self.unet.dice_loss,
            'tversky_loss': self.unet.tversky_loss,
            'weighted_dice_loss': self.unet.weighted_dice_loss,
            'combined_loss': self.unet.combined_loss,
            'tversky_loss_07_03': lambda y_true, y_pred: self.unet.tversky_loss(y_true, y_pred, 0.7, 0.3),
            'weighted_dice_w5': lambda y_true, y_pred: self.unet.weighted_dice_loss(y_true, y_pred, weight=5.0),
            'loss': self.unet.dice_loss
        }
        
        self.segmentation_model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objs
        )
        print(f"Модель загружена: {model_path}")
        return self.segmentation_model

    def load_best_model(self, info_path='model_train/best_model_info.json'):
        """
        Загружает лучшую модель и ее оптимальный порог из JSON-файла.
        """
        with open(info_path, 'r') as f:
            best_info = json.load(f)
            
        self.load_segmentation_model(f'model_train/{best_info["best_model_file"]}')
        self.best_threshold = best_info['best_threshold']
        
        print(f"Оптимальный порог для этой модели: {self.best_threshold:.2f}")
        return self.best_threshold

    def segment_image(self, image_path, output_path=None, patch_size=256, overlap=0.5, threshold=0.5):
        """
        Сегментация спутникового снимка с батчевой обработкой патчей.
        Использует patchify для самостоятельного разбиения на батчи
        и использует параллельные вычисления, собирает изображения.
        """
        with rasterio.open(image_path) as src:
            # Читаем RGB каналы (B4, B3, B2)
            red = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            blue = src.read(3).astype(np.float32)
            
            # Нормализация с percentile stretch
            def norm_band(band):
                p2, p98 = np.percentile(band, (2, 98))
                return np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
                
            img = np.stack([norm_band(red), norm_band(green), norm_band(blue)], axis=-1).astype(np.float32)
            h, w = img.shape[:2]
            
            step = int(patch_size * overlap) if overlap < 1 else patch_size // 2
            
            # Координаты левых верхних углов патчей
            y_coords = list(range(0, h - patch_size + 1, step))
            x_coords = list(range(0, w - patch_size + 1, step))
            n_patches = len(y_coords) * len(x_coords)
            
            # Собираем все патчи в один массив для батчевого предсказания
            patches = []
            for y in y_coords:
                for x in x_coords:
                    patch = img[y:y + patch_size, x:x + patch_size, :]
                    patches.append(patch)
                    
            patches = np.array(patches) # (n_patches, patch_size, patch_size, 3)
            
            # Батчевое предсказание
            print(f"Обработка {n_patches} патчей...")
            predictions = self.segmentation_model.predict(patches, batch_size=16, verbose=1)
            predictions = predictions[..., 0] # (n_patches, patch_size, patch_size)
            
            # Аккумуляция результатов с весами
            mask_accum = np.zeros((h, w), dtype=np.float32)
            weight_accum = np.zeros((h, w), dtype=np.float32)
            
            idx = 0
            for y in y_coords:
                for x in x_coords:
                    mask_accum[y:y + patch_size, x:x + patch_size] += predictions[idx]
                    weight_accum[y:y + patch_size, x:x + patch_size] += 1
                    idx += 1
                    
            # Усредняем перекрывающиеся области
            self.segmentation_masks = mask_accum / np.maximum(weight_accum, 1e-8)
            
            # Сохраняем бинарную маску с заданным порогом
            if output_path:
                binary = (self.segmentation_masks > threshold).astype(np.uint8)
                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=h,
                    width=w,
                    count=1,
                    dtype=np.uint8,
                    crs=src.crs,
                    transform=src.transform
                ) as dst:
                    dst.write(binary, 1)
                print(f"Маска сохранена: {output_path}")
                
            return self.segmentation_masks

    def mask_to_geojson(self, mask_path, output_path='water_bodies.geojson', min_area=1000):
        """
        Конвертация бинарной маски (GeoTIFF) в векторный слой GeoJSON.
        Фильтрует полигоны по минимальной площади (в квадратных метрах).
        
        Параметры:
        - mask_path: путь к GeoTIFF с бинарной маской (0 или 1).
        - output_path: путь для сохранения GeoJSON.
        - min_area: минимальная площадь полигона в кв. метрах.
        
        Возвращает:
        - gdf: GeoDataFrame с полигонами водных объектов.
        """
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs
            
        results = []
        # Извлекаем геометрию из маски
        for geom, value in features.shapes(mask.astype(np.int16), transform=transform):
            if value == 1:
                geom_shapely = shape(geom)
                if geom_shapely.area >= min_area:
                    results.append(geom_shapely)
                    
        gdf = gpd.GeoDataFrame(geometry=results, crs=crs)
        gdf = gdf.to_crs('EPSG:4326')
        gdf.to_file(output_path, driver='GeoJSON')
        
        print(f"GeoJSON сохранен: {output_path} (объектов: {len(gdf)})")
        return gdf

    def segment_and_export(self, image_path, output_mask='segmentation_mask.tif', 
                           output_geojson='water_bodies.geojson', threshold=0.5, min_area=1000):
        """
        Полный цикл: сегментация снимка и экспорт в GeoTIFF и GeoJSON.
        """
        self.segment_image(image_path, output_path=output_mask, threshold=threshold)
        # Можно пересохранить маску с другим порогом
        # Здесь используем уже сохраненный файл output_mask
        self.mask_to_geojson(output_mask, output_geojson, min_area=min_area)
        print("Интеграция завершена: получены растровая и векторная маски.")

    def save_pract34_results(self, models, results, X_train, X_val, X_test):
        """
        Сохранение результатов практической работы №3-4.
        
        Параметры:
        - models: словарь обученных моделей {имя: модель}
        - results: словарь с метриками моделей (от evaluate_models)
        - X_train, X_val, X_test: массивы данных (для статистики)
        """
        os.makedirs(self.train_dir, exist_ok=True)
        
        # 1. Сохраняем все модели
        for name, model in models.items():
            # Приводим имя к безопасному виду для имени файла
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            model_path = os.path.join(self.train_dir, f'{safe_name}.h5')
            model.save(model_path)
            print(f"Модель сохранена: {model_path}")
            
        # 2. Сохраняем результаты сравнения (метрики)
        results_path = os.path.join(self.train_dir, 'segmentation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Результаты сравнения сохранены: {results_path}")
        
        # 3. Находим лучшую модель (по максимальному IoU)
        best_model_name = max(results, key=lambda n: results[n]['IoU'])
        best_iou = results[best_model_name]['IoU']
        
        # 4. Метаданные эксперимента
        metadata = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'best_iou': float(best_iou), # преобразуем numpy float в обычный float
            'best_model': best_model_name,
            'tversky_params': {'alpha': 0.7, 'beta': 0.3},
            'date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = os.path.join(self.train_dir, 'pract34_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Метаданные сохранены: {metadata_path}")
        
        print("\nВсе файлы сохранены в папке:", self.train_dir)
        print(" - все обученные модели (*.h5)")
        print(" - segmentation_results.pkl")
        print(" - pract34_metadata.json")
        print(" - графики (loss_comparison.png, *_training.png, threshold_analysis_*.png и др.)")