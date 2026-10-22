import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
import os
import utm
import json

class WaterQualityPipeline:
    def __init__(self, name="Мониторинг качества воды", output_dir='class_data/'):
        self.name = name
        self.samples = None
        self.samples_sk42 = None
        self.world_data = None
        self.results = {}
        self.output_dir = output_dir
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Создан объект: {name}")


    def load_world_data(self, path='data/ne_110m_admin_0_countries.shp'):
        """Загрузка данных о странах мира"""
        self.world_data = gpd.read_file(path)
        print(f"Загружены данные о {len(self.world_data)} странах")
        return self.world_data


    def create_sample_points(self, n_points, region):
        """Создание тестовых точек проб воды"""
        if region == 'ekb':
            lats = np.random.uniform(56.5, 57.0, n_points)
            lons = np.random.uniform(60.0, 61.0, n_points)
        else:
            raise ValueError("Неизвестный регион")
        
        data = pd.DataFrame({
            'id': range(1, n_points + 1),
            'lat': lats,
            'lon': lons,
            'name': [f'Точка_{i}' for i in range(1, n_points + 1)]
        })
        geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        self.samples = gpd.GeoDataFrame(
            data,
            geometry=geometry,
            crs='EPSG:4326'
        )
        points_file = os.path.join(self.output_dir, f'sample_points_{region}.geojson')
        self.samples.to_file(points_file, driver='GeoJSON')
        print(f"Создано {len(self.samples)} тестовых точек проб")
        return self.samples

    def summary(self):
        """Краткая сводка о состоянии пайплайна"""

        print("\n" + "=" * 50)
        print(f"СОСТОЯНИЕ ПАЙПЛАЙНА: {self.name}")
        print("=" * 50)
        print(f"Пробы: {len(self.samples) if self.samples is not None else 'не созданы'}")
        print(f"CRS проб: {self.samples.crs if self.samples is not None else 'N/A'}")
        print(f"Данные мира: {'загружены' if self.world_data is not None else 'нет'}")
        print(f"Результаты: {len(self.results)} сохраненных результатов")
        print("=" * 50)

    def create_sk42_points(self, n_points=5):
            """Создание тестовых точек в системе СК-42 (зона 7)
            Координаты в СК-42 (метры) для территории Свердловской области"""
    
            x_coords = np.random.uniform(6269000, 6320000, n_points)
            y_coords = np.random.uniform(11310000, 11370000, n_points)
            data = pd.DataFrame({
                'id': range(1, n_points + 1),
                'x_sk42': x_coords,
                'y_sk42': y_coords,
                'name': [f'СК-42_точка_{i}' for i in range(1, n_points + 1)]
            })
            geometry = [Point(y, x) for y, x in zip(y_coords, x_coords)]
            self.samples_sk42 = gpd.GeoDataFrame(
                data,
                geometry=geometry,
                crs='EPSG:28408'
            )
            
            return self.samples_sk42

    def convert_to_wgs84(self):
        """ Преобразование проб в WGS84 """

        if self.samples_sk42 is None:
            raise ValueError("Нет данных для преобразования")

        original_crs = self.samples_sk42.crs
        self.samples = self.samples_sk42.to_crs('EPSG:4326')
        print(f"Координаты преобразованы из {original_crs} в WGS84")
        return self.samples

    def calculate_distances(self, gdf_sk42):
        """
        Расчет расстояний между точками в разных проекциях
        """
        gdf_wgs84 = gdf_sk42.to_crs('EPSG:4326')
        
        lon, lat = gdf_wgs84.geometry.x.mean(), gdf_wgs84.geometry.y.mean()
        zone_number = utm.from_latlon(lat, lon)[2]
        epsg_code = 32600 + zone_number
        gdf_utm = gdf_wgs84.to_crs(f'EPSG:{epsg_code}')
        print(f"Зона UTM: {zone_number}N (EPSG:{epsg_code})")

        distances_utm = []
        distances_deg = []

        for i in range(len(gdf_wgs84) - 1):
            p1_utm = gdf_utm.geometry.iloc[i]
            p2_utm = gdf_utm.geometry.iloc[i + 1]
            dist_utm = p1_utm.distance(p2_utm)
            distances_utm.append(dist_utm)
            p1_deg = gdf_wgs84.geometry.iloc[i]
            p2_deg = gdf_wgs84.geometry.iloc[i + 1]
            dist_deg = p1_deg.distance(p2_deg)
            distances_deg.append(dist_deg)
        
        distances_deg_km = [d * 111 for d in distances_deg]

        print("\nСравнение расстояний:")
        print("-" * 50)
        print("Пара | В UTM (м) | В градусах (км, грубо) | Разница")
        print("-" * 50)
        for i, (d_utm, d_deg_km) in enumerate(zip(distances_utm, distances_deg_km)):
            diff = abs(d_utm / 1000 - d_deg_km)
            print(f"{i + 1}-{i + 2} | {d_utm:.0f} м | {d_deg_km:.2f} км | {diff:.2f} км")

    def compare_area_calculations(self):
            if self.world_data is None:
                raise ValueError("Сначала загрузите данные о странах (load_world_data)")
            
            russia = self.world_data[self.world_data['NAME'] =='Russia'].copy()
    
            print("\nСРАВНЕНИЕ РАСЧЁТА ПЛОЩАДИ РОССИИ")
            print("=" * 60)
    
            area_deg = russia.geometry.area.iloc[0]
            area_deg_km2 = area_deg * (111.32 * 111.32)
    
            print(f"В WGS84 (неправильно!): {area_deg_km2 / 1e6:.2f} млн км²")
    
            russia_moll = russia.to_crs('ESRI:54009')
            area_moll = russia_moll.geometry.area.iloc[0] / 1e12
    
            print(f"Равновеликая (Мольвейде): {area_moll:.2f} млн км²")
    
            russia_aea = russia.to_crs('ESRI:102013')
            area_aea = russia_aea.geometry.area.iloc[0] / 1e12
    
            print(f"Альберса (для России): {area_aea:.2f} млн км²")
    
            real_area = 17.13
    
            print("-" * 60)
            print(f"Реальная площадь России: {real_area} млн км²")
            print(f"Ошибка WGS84: {abs(area_deg_km2 / 1e6 - real_area) / real_area * 100:.1f}%")
    
            results = {
                'wgs84': area_deg_km2 / 1e6,
                'mollweide': area_moll,
                'aea': area_aea,
                'real': real_area
            }
            self.results['area_comparison'] = results
    
            return results

    def plot_area_comparison(self):
            if 'area_comparison' not in self.results:
                raise ValueError("Сначала выполните compare_area_calculations()")
            
            areas = self.results['area_comparison']
            fig, ax = plt.subplots(figsize=(10, 6))
            methods = ['WGS84 (ошибочно)', 'Мольвейде', 'Альберса', 'Реальная']
            values = [areas['wgs84'], areas['mollweide'],
            areas['aea'], areas['real']]
            colors = ['red', 'green', 'green', 'blue']
            bars = ax.bar(methods, values, color=colors, alpha=0.7)
            ax.set_ylabel('Площадь (млн км²)')
            ax.set_title('Сравнение расчета площади России в разныхпроекциях')
            ax.axhline(y=areas['real'], color='blue', linestyle='--',
            label='Реальная площадь')
            ax.legend()
    
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 
                        0.2, f'{val:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('area_comparison.png', dpi=150)
            plt.show()
            print("График сохранен: area_comparison.png")

    def visualize_samples(self, filename='samples_map.html'):
        if self.samples is None:
            raise ValueError("Нет данных для визуализации")

        center_lat = self.samples.geometry.y.mean()
        center_lon = self.samples.geometry.x.mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

        sw = self.samples.geometry.y.min(), self.samples.geometry.x.min()
        ne = self.samples.geometry.y.max(), self.samples.geometry.x.max()

        m.fit_bounds([sw, ne])

        m.fit_bounds([sw, ne], padding=(30, 30))

        for idx, row in self.samples.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=f"ID: {row['id']}<br>Название: {row['name']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник'
        ).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(self.output_dir+filename)
        print(f"Карта сохранена: {self.output_dir + filename}")

        return m

    def create_buffer_zones(self, buffer_size_km=4, filename='buffer_zones_map.html'):
            if self.samples is None:
                raise ValueError("Нет данных")
            
            lon, lat = self.samples.geometry.x.mean(), self.samples.geometry.y.mean()
            epsg_code = 32600 + utm.from_latlon(lat, lon)[2]
            samples_utm = self.samples.to_crs(f'EPSG:{epsg_code}')
            buffer_size_m = buffer_size_km * 1000
            samples_utm['buffer'] = samples_utm.geometry.buffer(buffer_size_m)
            
            buffers_gdf = gpd.GeoDataFrame(
                samples_utm[['id', 'name']],
                geometry=samples_utm['buffer'],
                crs=samples_utm.crs
            )
    
            buffers_wgs84 = buffers_gdf.to_crs('EPSG:4326')
            center_lat = self.samples.geometry.y.mean()
            center_lon = self.samples.geometry.x.mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
            folium.GeoJson(
                buffers_wgs84,
                name='Буферные зоны',
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'blue',
                    'weight': 1,
                    'fillOpacity': 0.1
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['id', 'name'],
                    aliases=['ID точки:', 'Название:'],
                    localize=True
                )
            ).add_to(m)
    
            for idx, row in self.samples.iterrows():
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=f"ID: {row['id']}<br>Название: {row['name']}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
    
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Спутник'
            ).add_to(m)
            folium.LayerControl().add_to(m)
            m.save(self.output_dir + filename)
            print(f"Карта с буферными зонами сохранена: {self.output_dir + filename}")
    
            return filename

    def save_results(self):
        results = {
            'name': self.name,
            'n_samples': len(self.samples) if self.samples is not None else 0,
            'crs': str(self.samples.crs) if self.samples is not None else None,
            'stats': {
                'mean_lat': float(self.samples.geometry.y.mean()) if self.samples is not None else 0,
                'mean_lon': float(self.samples.geometry.x.mean()) if self.samples is not None else 0,
            }
        }
        with open(self.output_dir + '1_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

            if self.samples is not None:
                self.samples.to_file(self.output_dir + 'work1_samples.geojson', driver='GeoJSON')

            print("Результаты сохранены:")
            print(" - work1_results.json")
            print(" - work1_samples.geojson")
            print(" - work1_samples_map.html")
            print(" - buffer_zones_map.html")
            print(" - area_comparison.png")