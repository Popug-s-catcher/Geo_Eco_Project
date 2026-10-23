from src.WaterQualityPipelineSegExt import WaterQualityPipelineSegExt

def task_2_1(pipeline: WaterQualityPipelineSegExt):
    pipeline.test_methods_on_patches(n_samples=3)

def task_3_3(pipeline: WaterQualityPipelineSegExt):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.load_dataset()
    pipeline.visualize_augmentation(n_samples=3)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

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

def lb3_run():
    pipeline = WaterQualityPipelineSegExt()
    #pipeline.create_dataset_from_pract2(stride=64)

    task_2_1(pipeline)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = task_3_3(pipeline)
    task_4_4(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
