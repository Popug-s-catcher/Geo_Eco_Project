from src.WaterQualityPipelineSegExt import WaterQualityPipelineSegExt
from src.SegmentationPipeline import SegmentationPipeline
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def task_2_1(pipeline: WaterQualityPipelineSegExt):
    pipeline.test_methods_on_patches(n_samples=3)

def task_3_3(pipeline: WaterQualityPipelineSegExt):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.load_dataset()
    pipeline.visualize_augmentation(n_samples=3)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# !!! U-NET в другом классе !!!
def task_4_4(
        pipeline: WaterQualityPipelineSegExt,
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test
        ):
    # Этап 4: Демонстрация U-Net и обучение
    unet_model = pipeline.demonstrate_unet()
    unet_model.save('model_train/unet_model.h5')
    
    models, histories, results = pipeline.compare_loss_functions(
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test, 
        epochs=50
    )

    return models, histories, results

def task_5_1(
        pipeline: WaterQualityPipelineSegExt, 
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
    # Путь к новому снимку
    test_image = 'имя_нового_снимка'
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

def task_6_2(
        pipeline: WaterQualityPipelineSegExt,
        models, results, 
        X_train, X_val, X_test
    ):

    # Сохранение результатов
    pipeline.save_pract34_results(models, results, X_train, X_val, X_test)

def lb3_run():
    pipeline = WaterQualityPipelineSegExt()
    seg_pipeline = SegmentationPipeline()
    #pipeline.create_dataset_from_pract2(stride=64)

    optimal_thresholds = {}     # оптимальный порог для каждой модели
    val_ious_optimal = {}       # IoU на валидации с оптимальным порогом

    task_2_1(pipeline)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = task_3_3(pipeline)
    #models = task_4_4(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
    #task_6_2(pipeline, models, results, X_train, X_val, X_test)                    # Сохранение результатов
    #task_5_1(pipeline, models, X_val, y_val,optimal_thresholds, val_ious_optimal)
    #task_5_2(pipeline, X_train, y_train, X_val, y_val)
    #task_5_3(pipeline, X_test, y_test, models, optimal_thresholds)
    #task_6_1(seg_pipeline)
