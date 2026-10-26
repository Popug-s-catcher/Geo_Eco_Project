import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.exposure import equalize_hist
from skimage.morphology import remove_small_objects, closing, disk

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

    def global_threshold(self, image, threshold=0.5):
        """
        Глобальный порог по NDWI
        """
        ndwi = self._get_ndwi(image)
        return (ndwi > threshold).astype(np.float32) if ndwi is not None else np.zeros(image.shape[:2])


    def otsu_threshold(self, image):
        """
        Метод ОЦУ на NDWI
        Предварительно улучшает контраст (equalize_hist)
        и автоматически находит оптимальный порог
        """
        ndwi = self._get_ndwi(image)
        if ndwi is None:
            return np.zeros(image.shape[:2])

        ndwi_pos = np.clip(ndwi, 0, None)
        if ndwi_pos.max() == 0:
            return np.zeros(ndwi.shape)

        ndwi_eq = equalize_hist(ndwi_pos)
        try:
            return (ndwi_eq > threshold_otsu(ndwi_eq)).astype(np.float32)
        except:
            return (ndwi > np.median(ndwi)).astype(np.float32)


    def adaptive_threshold(self, image, block_size=101, offset=-0.2):
        """
        Адаптивный порог по NDWI: свой порог для каждой области изображения
        block_size - размер окна (нечетный), offset - смещение порога
        """
        ndwi = self._get_ndwi(image)
        if ndwi is None:
            return np.zeros(image.shape[:2])

        local_thresh = threshold_local(ndwi, block_size, offset=offset)
        return (ndwi > local_thresh).astype(np.float32)


    def ndwi_threshold(self, image_4band, threshold=0.15):
        """
        NDWI порог для 4-канальных данных
        Без канала NIR использует grayscale: вода темнее суши, порог - метод Оцу
        """
        if image_4band.ndim == 3 and 4 in (image_4band.shape[0], image_4band.shape[-1]):
            ndwi = self._get_ndwi(image_4band)
            return (ndwi > threshold).astype(np.float32)

        gray = image_4band.mean(axis=-1) if image_4band.ndim == 3 else image_4band
        gray = gray.astype(np.float32)

        if gray.max() == gray.min():
            return np.zeros(gray.shape, dtype=np.float32)
        
        return (gray < threshold_otsu(gray)).astype(np.float32)

    def postprocess_mask(self, mask, min_size=50):
        """
        в class WaterSegmentation
        Постобработка бинарной маски
        Удаляет маленькие объекты и заполняет дыры
        """

        mask_bool = mask > 0.5
        mask_clean = remove_small_objects(mask_bool, max_size=min_size - 1)
        return closing(mask_clean, disk(2)).astype(np.float32)
    
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