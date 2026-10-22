import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk

class WaterSegmentation:
    # Класс для сегментации водных объектов
    def __init__(self):
        self.results = {}
    
    def _get_ndwi(self, image):
        """
        Универсальное вычисление NDWI из любых данных
        NDWI = (Green - NIR) / (Green + NIR) для 4-канальных
        данных
        или (Green - Blue) / (Green + Blue) для RGB
        """
        if hasattr(image, 'shape') and len(image.shape) == 3:
            if image.shape[-1] == 4: # RGBNIR (каналы в конце)
                green, nir = image[:, :, 1].astype(np.float32), image[:, :, 3].astype(np.float32)
                return (green - nir) / (green + nir + 1e-8)
            elif image.shape[0] == 4: # RGBNIR (каналы первыми)
                green, nir = image[1].astype(np.float32), image[3].astype(np.float32)
                return (green - nir) / (green + nir + 1e-8)
            elif image.shape[-1] >= 3: # RGB
                green, blue = image[:, :, 1].astype(np.float32), image[:, :, 2].astype(np.float32)
                return (green - blue) / (green + blue + 1e-8)
            return None

    def global_threshold(self, image, threshold=0.15):
        """
        Глобальный порог по NDWI
        """
        ndwi = self._get_ndwi(image)
        return (ndwi > threshold).astype(np.float32) if ndwi is not None else np.zeros(image.shape[:2])

    def otsu_threshold(self, image):
        """
        Метод ОЦУ на NDWI
        Автоматически находит оптимальный порог
        """
        ndwi = self._get_ndwi(image)
        if ndwi is None:
            return np.zeros(image.shape[:2])
    
        ndwi_pos = np.clip(ndwi, 0, None)
    
        if ndwi_pos.max() == 0:
            return np.zeros(ndwi.shape)
        
        ndwi_norm = ((ndwi_pos - ndwi_pos.min()) / (ndwi_pos.max() - ndwi_pos.min()) * 255).astype(np.uint8)
        
        try:
            thresh = threshold_otsu(ndwi_norm) / 255
            return (ndwi > thresh).astype(np.float32)
        except:
            return (ndwi > np.median(ndwi)).astype(np.float32)

    def adaptive_threshold(self, image, threshold=0.1):
        """
        Адаптивный порог - упрощенная версия для NDWI
        """
        ndwi = self._get_ndwi(image)
        return (ndwi > threshold).astype(np.float32) if ndwi is not None else np.zeros(image.shape[:2])

    def ndwi_threshold(self, image_4band, threshold=0.15):
        """
        NDWI порог для 4-канальных данных
        """
        if image_4band.shape[0] != 4:
            return np.zeros((image_4band.shape[1], image_4band.shape[2]))
        
        green, nir = image_4band[1].astype(np.float32), image_4band[3].astype(np.float32)
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        return (ndwi > threshold).astype(np.float32)

    def postprocess_mask(self, mask, min_size=50):
        """
        в class WaterSegmentation
        Постобработка бинарной маски
        Удаляет маленькие объекты и заполняет дыры
        """

        mask_bool = mask > 0.5
        mask_clean = remove_small_objects(mask_bool,min_size=min_size)
        return binary_closing(mask_clean, disk(2)).astype(np.float32)
    
    def calculate_metrics(self, true_mask, pred_mask):
        """
        в class WaterSegmentation
        Расчет метрик качества сегментации
        IoU (Intersection over Union) - основная метрика
        """
        
        true_bin, pred_bin = true_mask > 0.5, pred_mask > 0.3
        tp = np.sum(true_bin & pred_bin)
        fp = np.sum(~true_bin & pred_bin)
        fn = np.sum(true_bin & ~pred_bin)

        return {'IoU': tp / (tp + fp + fn + 1e-10), 'TP': tp, 'FP': fp, 'FN': fn}