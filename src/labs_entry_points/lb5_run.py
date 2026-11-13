from src.WaterQualityPipelineAnalytics import WaterQualityPipelineAnalytics
import os


def task_1_1(analyzer: WaterQualityPipelineAnalytics):
    analyzer.load_data()
    analyzer.show_pdk()
    analyzer.check_data()


def task_2_1(analyzer: WaterQualityPipelineAnalytics):
    analyzer.calculate_izv_all()
    analyzer.show_izv_results()


def task_2_2(analyzer: WaterQualityPipelineAnalytics):
    analyzer.convert_to_wgs84()
    analyzer.create_izv_map()
    analyzer.create_static_izv_map()


def task_2_3(analyzer: WaterQualityPipelineAnalytics):
    analyzer.plot_pollutant_histograms()


def task_3_1(analyzer: WaterQualityPipelineAnalytics):
    analyzer.calculate_chlorophyll()
    analyzer.plot_chlorophyll_analysis()


def task_3_2(analyzer: WaterQualityPipelineAnalytics):
    analyzer.show_chlorophyll_statistics()
    analyzer.show_trophic_status()


def task_4_1():
    os.makedirs('templates', exist_ok=True)
    os.makedirs('reports', exist_ok=True)


def task_5_1(analyzer: WaterQualityPipelineAnalytics):
    chlorophyll_data = analyzer.collect_chlorophyll_data()
    analyzer.prepare_report_data(chlorophyll_data)
    analyzer.show_report_summary()


def task_5_2(analyzer: WaterQualityPipelineAnalytics):
    analyzer.generate_html_report()


def task_5_3(analyzer: WaterQualityPipelineAnalytics):
    analyzer.generate_pdf_report()


def task_4_4(analyzer: WaterQualityPipelineAnalytics):
    for lang in ['ru', 'en']:
        analyzer.generate_report_figures(lang=lang)
        analyzer.generate_html_report(lang=lang)
        analyzer.generate_pdf_report(lang=lang)


def task_5_4(analyzer: WaterQualityPipelineAnalytics):
    analyzer.generate_docx_report()


def task_6_1(analyzer: WaterQualityPipelineAnalytics):
    analyzer.prepare_gis_export()
    analyzer.export_geojson()


def task_6_2(analyzer: WaterQualityPipelineAnalytics):
    analyzer.export_csv()


def task_7_1(analyzer: WaterQualityPipelineAnalytics):
    analyzer.generate_report()


def task_2_4(analyzer: WaterQualityPipelineAnalytics):
    analyzer.calculate_contributions()
    analyzer.show_contribution_statistics()
    analyzer.create_contribution_map()


def lb5_run():
    analyzer = WaterQualityPipelineAnalytics()
    task_1_1(analyzer)
    # task_2_1(analyzer)
    # task_2_2(analyzer)
    # task_2_3(analyzer)
    # task_2_4(analyzer)
    # task_3_1(analyzer)
    # task_3_2(analyzer)
    # task_4_1()
    # task_5_1(analyzer)
    # task_5_2(analyzer)
    # task_5_3(analyzer)
    # task_4_4(analyzer)
    # task_5_4(analyzer)
    # task_6_1(analyzer)
    # task_6_2(analyzer)
    # task_7_1(analyzer)