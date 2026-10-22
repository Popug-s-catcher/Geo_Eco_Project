import geopandas as gpd
import shapely
import pyproj
import folium
import matplotlib.pyplot as plt
from src.WaterQualityPipelineExtended import WaterQualityPipelineExtended
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import webbrowser
import ee

def load_sputnik_image():
    # 1. Инициализация Earth Engine
    ee.Authenticate()
    ee.Initialize(project='pugs-512215')

    # 2. Определяем область интереса (Екатеринбург)
    # Можно задать область прямоугольником
    # lat1 56.64 lon1 60
    # lat2 56.84 lon2 60.2
    roi = ee.Geometry.Rectangle([60, 56.64, 60.2, 56.79])
    # Можно задать через конкретные точки
    # data = {
    #     'name': ['Point 1', 'Point 2', 'Point 3'],
    #     'lon': [30.204035, 30.358192, 30.244896],
    #     'lat': [59.985333, 59.978309, 59.909067]
    # }

    # # Из табличных координат в геопространственный объект
    # gdf = gpd.GeoDataFrame(
    #     data,
    #     geometry=gpd.points_from_xy(data['lon'], data['lat']),
    #     crs="EPSG:4326"
    # )

    # Создаем ROI как прямоугольник по границам точек
    # min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    # roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # 3. Поиск снимка Sentinel-2 (лето 2025, облачность < 20%)
    collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate('2025-06-01', '2025-08-01')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )

    # 4. Проверка наличия снимков
    size = collection.size().getInfo()
    print(f"Найдено снимков: {size}")
    if size > 0:
        image = collection.first()
        cloudiness = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        print(f"Облачность лучшего снимка: {cloudiness}%")

        # Выбираем каналы: B4(Red), B3(Green), B2(Blue), B8(NIR), B11(SWIR)
        selected_bands = image.select(['B4', 'B3', 'B2', 'B8', 'B11'])

        # Экспорт в Google Drive
        task = ee.batch.Export.image.toDrive(
            image=selected_bands,
            description='sentinel2_spb',
            folder='gee_exports',
            fileNamePrefix='sentinel2_16bit_data',
            region=roi,
            scale=10,
            fileFormat='GeoTIFF',
            maxPixels=1e9
        )

        task.start()
        print("Задача экспорта запущена!")
        print("После завершения скачайте снимок в папку data/")


def task_1_2(pipeline: WaterQualityPipelineExtended):
    path_to_file = 'data/degtyarsk_water.tif'
    pipeline.load_raster(path_to_file)

def task_2_1(pipeline: WaterQualityPipelineExtended):
    indices = pipeline.calculate_all_indices()
    for name, values in indices.items():
        print(f'{name}: min = {values.min():.3f}, max = {values.max():.3f}')

def task_3_1(pipeline: WaterQualityPipelineExtended):
    pipeline.create_water_sample_points(n_points=50)
    pipeline.points_gdf = gpd.read_file('data/water_points.geojson')

def task_3_2(pipeline: WaterQualityPipelineExtended):
    pipeline.plot_with_points()

def task_4_1(pipeline: WaterQualityPipelineExtended):
    if (pipeline.points_gdf is None) or len(pipeline.points_gdf) == 0:
        raise ValueError("points_gdf должен быть заполнен данными")
      
    features = ['NH4_mgL', 'PO4_mgL', 'pH']
    X = pipeline.points_gdf[features].values
    X_scaled = StandardScaler().fit_transform(X)
    optimal_eps = pipeline.find_optimal_eps_auto(X_scaled, min_samples=4)
    pipeline.clusters = DBSCAN(eps=optimal_eps, min_samples=4).fit_predict(X_scaled)
    pipeline.points_gdf['cluster'] = pipeline.clusters
    n_clusters = len(set(pipeline.clusters)) - (1 if -1 in pipeline.clusters else 0)
    print(f"Найдено {n_clusters} кластеров (eps={optimal_eps:.3f})")

def task_5(pipeline: WaterQualityPipelineExtended):
    pipeline.visualize_clusters_map(filename='class_data/clusters_map.html')
    os.makedirs('class_data', exist_ok=True)
    pipeline.points_gdf.to_file('class_data/points_with_clusters.geojson', driver='GeoJSON')
    print("\nТочки с кластерами сохранены: class_data/points_with_clusters.geojson")

def task_6_1(pipeline: WaterQualityPipelineExtended):
    pipeline.analyze_ndvi_anomalies(n_years=10, seed=42)
    print("\nИТОГОВАЯ СТАТИСТИКА")
    print("="*50)
    print(f"Всего точек проб: {len(pipeline.points_gdf)}")
    print(f"Средний NDVI: {pipeline.indices['NDVI'].mean():.3f}")
    print(f"Средний NDWI: {pipeline.indices['NDWI'].mean():.3f}")
    print(f"Средний NDBI: {pipeline.indices['NDBI'].mean():.3f}")
    print(f"Диапазон NH4: {pipeline.points_gdf['NH4_mgL'].min():.2f} - {pipeline.points_gdf['NH4_mgL'].max():.2f} мг/л")

def lb2_run():
    #load_sputnik_image()
    pipeline = WaterQualityPipelineExtended()
    task_1_2(pipeline)
    task_2_1(pipeline)
    task_3_1(pipeline)
    task_3_2(pipeline)
    task_4_1(pipeline)
    task_5(pipeline)
    task_6_1(pipeline)