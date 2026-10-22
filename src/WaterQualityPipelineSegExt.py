from src.WaterQualityPipelineExtended import WaterQualityPipelineExtended
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import warnings
import os
import numpy as np
from PIL import Image
import rasterio

class WaterQualityPipelineSegExt(WaterQualityPipelineExtended): 
    def __init__(self, name="Сегментация"): 
        """Конструктор класса""" 
        super().__init__(name) 
        self.name = name 
        self.dataset_dir = 'water_dataset' # Папка с датасетом 
        self.train_dir = 'model_train' # Папка для сохранения моделей

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

