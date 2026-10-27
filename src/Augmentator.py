import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

class Augmentator():
    def __init__(self): 
            self.dataset_dir = 'water_dataset'      # Папка с датасетом
    
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