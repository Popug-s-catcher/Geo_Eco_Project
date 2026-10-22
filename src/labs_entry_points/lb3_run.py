from src.WaterQualityPipelineSegExt import WaterQualityPipelineSegExt


def lb3_run():
    pipeline = WaterQualityPipelineSegExt()

    pipeline.create_dataset_from_pract2(stride=64)
