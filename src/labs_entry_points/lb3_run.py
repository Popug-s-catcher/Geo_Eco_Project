from src.WaterQualityPipelineSegExt import WaterQualityPipelineSegExt
from src.SegmentationPipeline import SegmentationPipeline
from src.Unet import Unet
from src.Augmentator import Augmentator
import tensorflow as tf
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
import json

MODELS = {
    'Dice Loss': 'unet_dice.h5',
    'Tversky Loss': 'unet_tversky.h5',
    'Weighted Dice': 'unet_weighted_dice_5.h5',
    'Combined Loss': 'combined_loss.h5'
}


def task_2_1(pipeline: WaterQualityPipelineSegExt):
    pipeline.test_methods_on_patches(n_samples=3)

def task_3_2(pipeline: WaterQualityPipelineSegExt):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.load_dataset()
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def task_3_3(augmentator: Augmentator):
    augmentator.visualize_augmentation(n_samples=3)

def task_4_4(
        unet: Unet,
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test
        ):
    # Этап 4: Демонстрация U-Net и обучение
    unet_model = unet.demonstrate_unet()
    unet_model.save('model_train/unet_model.h5')
    
    models, histories, results = unet.compare_loss_functions(
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test, 
        epochs=40
    )

    return models, histories, results

def task_5_1(
        pipeline: SegmentationPipeline, 
        models, 
        X_val, y_val,
        optimal_thresholds,
        val_ious_optimal
    ):
    
    for model_name, model in models.items():
        # Поиск оптимального порога
        # Возвращаем оптимальный порог, максимальное значение IoU
        opt_thresh, best_iou_val = pipeline.find_optimal_threshold(
            model, X_val, y_val, model_name=model_name,
            thresholds=np.arange(0.3, 0.85, 0.05)       # Перебираем пороги бинаризации
        )
        optimal_thresholds[model_name] = opt_thresh
        val_ious_optimal[model_name] = best_iou_val

def task_5_2(
        pipeline: WaterQualityPipelineSegExt,
        X_train, y_train,
        X_val, y_val
        ):
    
    tversky_results = pipeline.experiment_tversky_params(
        X_train, y_train, X_val, y_val,
        param_grid=[(0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)],
        epochs=20,      # меньше эпох для экономии времени
        sample_size=100 # используем подвыборку для ускорения
    )

    return tversky_results

def task_5_3(
        pipeline: WaterQualityPipelineSegExt,
        X_test, y_test,
        models,
        optimal_thresholds          
        ):
    
    pipeline.visualize_predictions(
        models, X_test, y_test,
        n_samples=4,
        save=True
    )

    # Визуализация с оптимальными порогами
    pipeline.visualize_predictions_with_thresholds(
        models, X_test, y_test,
        optimal_thresholds=optimal_thresholds, 
        n_samples=4
    )

def task_6_1(seg_pipeline: SegmentationPipeline):

    info_path = f'{seg_pipeline.train_dir}/best_model_info.json'

    if not os.path.exists(info_path):
        print("Файл best_model_info.json не найден. Запускаем поиск лучшей модели...")
        seg_pipeline.load_dataset()
        X_val, y_val = seg_pipeline.dataset['X_val'], seg_pipeline.dataset['y_val']
        best = None
        
        for file in MODELS.values():
            # ИСПРАВЛЕНО: убрано добавление .keras, так как в MODELS уже указано .h5
            model_path = f'{seg_pipeline.train_dir}/{file}'
            
            if not os.path.exists(model_path):
                print(f"Внимание: Файл {model_path} не найден, пропускаем.")
                continue
                
            print(f"Анализ модели: {file}")
            model = tf.keras.models.load_model(model_path, compile=False)
            threshold, iou = seg_pipeline.find_optimal_threshold(
                model, X_val, y_val, 
                model_name=file, 
                thresholds=np.arange(0.3, 0.85, 0.05)
            )
            
            if best is None or iou > best[2]:
                best = (file, threshold, iou)
        
        if best is not None:
            # ИСПРАВЛЕНО: сохраняем имя файла как есть (best[0]), без добавления .keras
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'best_model_file': best[0], 
                    'best_threshold': round(float(best[1]), 2),
                    'val_iou': round(float(best[2]), 4)
                }, f, ensure_ascii=False, indent=2)
            print(f"Информация о лучшей модели сохранена: {best[0]} (IoU: {best[2]:.4f})")
        else:
            raise RuntimeError("Ни одна модель не была найдена в папке model_train!")

    seg_pipeline.load_best_model(info_path)

    
    # Путь к новому снимку
    test_image = 'data/degtyarsk_water.tif'
    output_mask = 'data/full_mask.tif'
    output_geojson = 'data/water_bodies.geojson'
    
    # Применяем к тестовому снимку test_image
    seg_pipeline.segment_image(
        image_path=test_image,
        output_path=output_mask,
        threshold=seg_pipeline.best_threshold # используем оптимальный порог
    )

    # Конвертируем маску в GeoJSON
    seg_pipeline.mask_to_geojson(
        mask_path=output_mask,
        output_path=output_geojson,
        min_area=1000
    )

    # Визуализация результата
    with rasterio.open(output_mask) as src:
        mask = src.read(1)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='Blues')
    plt.title('Сегментированная маска всего снимка')
    plt.axis('off')
    plt.show()

def task_6_2(seg_pipeline: SegmentationPipeline, models, 
        X_train, X_val, X_test, y_val,
        thresholds, val_ious_optimal):

    
    task_5_1(seg_pipeline, models, X_val, y_val, thresholds, val_ious_optimal)
    X_test, y_test = seg_pipeline.dataset['X_test'], seg_pipeline.dataset['y_test']

    results = {}
    for name, model in models.items():
        
        probs = model.predict(X_test, batch_size=4, verbose=0)[..., 0]
        
        metrics = [seg_pipeline.segmenter.calculate_metrics(
                       y_test[i], seg_pipeline.segmenter.postprocess_mask((probs[i] > thresholds[name]).astype(np.float32)))
                   for i in range(len(y_test))]
        
        results[name] = {'IoU': float(np.mean([m['IoU'] for m in metrics])),
                         'threshold': round(float(thresholds[name]), 2)}
        
        print(f"{name:15}: IoU на test = {results[name]['IoU']:.4f} (порог {results[name]['threshold']:.2f})")

    seg_pipeline.save_pract34_results(models, results,
                                      seg_pipeline.dataset['X_train'], seg_pipeline.dataset['X_val'], X_test)

    seg_pipeline.save_pract34_results(models, results, X_train, X_val, X_test)


def lb3_run():
    pipeline = WaterQualityPipelineSegExt()
    seg_pipeline = SegmentationPipeline()
    unet = Unet()
    augmentator = Augmentator()
    #pipeline.create_dataset_from_pract2(stride=64)

    optimal_thresholds = {}     # оптимальный порог для каждой модели
    val_ious_optimal = {}       # IoU на валидации с оптимальным порогом

    task_2_1(pipeline)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = task_3_2(pipeline)
    #task_3_3(augmentator)
    #models = task_4_4(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
    #task_6_2(seg_pipeline, models, X_train, X_val, X_test, y_val)
    #task_5_1(pipeline, models, X_val, y_val, optimal_thresholds, val_ious_optimal)
    #task_5_2(pipeline, X_train, y_train, X_val, y_val)
    #task_5_3(pipeline, X_test, y_test, models, optimal_thresholds)
    #task_6_1(seg_pipeline)
