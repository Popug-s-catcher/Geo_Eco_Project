from src.WaterQualityPipelineSegExt import WaterQualityPipelineSegExt

def task_2_1(pipeline: WaterQualityPipelineSegExt):
    pipeline.test_methods_on_patches(n_samples=3)

def lb3_run():
    pipeline = WaterQualityPipelineSegExt()
    pipeline.create_dataset_from_pract2(stride=64)

    task_2_1(pipeline)
