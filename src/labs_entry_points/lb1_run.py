import geopandas as gpd
import shapely
import pyproj
import folium
import matplotlib.pyplot as plt
from src.WaterQualityPipeline import WaterQualityPipeline
import os
import webbrowser

def task_1_1():
    world = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    print("Данные загружены!")
    print(f"Количество стран: {len(world)}")
    print(f"Колонки: {world.columns.tolist()}")
    print(f"Система координат: {world.crs}")
    print(f"Типы геометрий: {world.geometry.geom_type.value_counts()}")
    return world

def task_1_2(world):
    print(world[['NAME', 'POP_EST', 'GDP_MD', 'geometry']].head())
    russia = world[world['NAME'] == 'Russia']
    print(f"\nРоссия: {russia.geometry.iloc[0].geom_type}")
    print(f"Граница России -- это {len(russia.geometry.iloc[0].geoms)} полигонов")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(ax=ax, color='lightgreen', edgecolor='black')
    plt.title('Карта мира')
    plt.savefig('world_map.png', dpi=150)
    plt.show()

def task_1_3(): 
    pipeline = WaterQualityPipeline("Практическая работа №1")
    pipeline.load_world_data('data/ne_110m_admin_0_countries.shp')
    pipeline.create_sample_points(15, region='ekb')
    pipeline.summary()
    return pipeline

def task_2_1(pipeline: WaterQualityPipeline):
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ С СК-42")
    print("="*50)
    sk42_points = pipeline.create_sk42_points(5)
    print("Точки в СК-42 созданы")
    print(sk42_points)
    points = pipeline.convert_to_wgs84()
    print(points)
    return sk42_points

def task_2_2(pipeline: WaterQualityPipeline, sk42_points: gpd.GeoDataFrame):
    pipeline.calculate_distances(sk42_points)

def task_3_1(pipeline: WaterQualityPipeline):
    print("\n" + "="*50)
    print("ИССЛЕДОВАНИЕ ПЛОЩАДЕЙ")
    print("="*50)
    pipeline.compare_area_calculations()
    pipeline.plot_area_comparison()

def task_4_1(pipeline: WaterQualityPipeline):
    region_name = 'ekb'
    pipeline.create_sample_points(20, region_name)
    output_file = f'map_{region_name}.html'
    full_path = pipeline.output_dir + output_file
    pipeline.visualize_samples(output_file)
    webbrowser.open('file://' + os.path.realpath(full_path))

def task_4_2(pipeline: WaterQualityPipeline):
    saved_file = pipeline.create_buffer_zones(buffer_size_km=4)
    full_path = pipeline.output_dir + saved_file
    webbrowser.open('file://' + os.path.realpath(full_path))

def task_5_1(pipeline: WaterQualityPipeline):
    pipeline.save_results()
    pipeline.summary()


def lb1_run():
    world = task_1_1()
    task_1_2(world)
    pipeline = task_1_3()
    sk42_points = task_2_1(pipeline)
    task_2_2(pipeline, sk42_points)
    task_3_1(pipeline)
    task_4_1(pipeline)
    task_4_2(pipeline)
    task_5_1(pipeline)