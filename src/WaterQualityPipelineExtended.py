from src.WaterQualityPipeline import WaterQualityPipeline
import rasterio
from rasterio.io import DatasetReader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import folium
import webbrowser
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import pandas as pd
from statsmodels.tsa.seasonal import STL
import random

class WaterQualityPipelineExtended(WaterQualityPipeline):
    def __init__(self, name="Практическая работа №2"):
        super().__init__(name)
        self.raster: DatasetReader | None = None
        self.bands: np.ndarray | None = None
        self.indices: dict[str, np.ndarray] = {}
        self.clusters: np.ndarray | None = None
        self.points_gdf: gpd.GeoDataFrame | None = None


    def load_raster(self, raster_path: str):
        """Загрузка спутникового снимка"""

        self.raster = rasterio.open(raster_path)

        if self.raster is None:
            raise FileExistsError("Неудалось открыть")

        self.bands = self.raster.read()
        
        print(f"Снимок загружен: {raster_path}")
        print(f"Размер: {self.raster.width} x {self.raster.height}")
        print(f"Каналов: {self.raster.count}")
        print(f"CRS: {self.raster.crs}")

        return self.raster

    def create_water_sample_points(self, n_points=50, output_file='data/water_points.geojson'):
        """
        Создание точек проб с 4 заданными кластерами
        Алгоритм:
        1. Делит область снимка на 4 части по координатам
        2. В каждой части создает точки с разными диапазонами NH4, PO4, pH
        3. Формирует 4 кластера для демонстрации DBSCAN
        Параметры кластеров (гауссовы распределения):
        - Кластер 0 (чистая вода): NH4=0.05-0.25, PO4=0.02-0.10, pH=7.8-8.3
        - Кластер 1 (органика): NH4=1.2-2.2, PO4=0.25-0.55, pH=6.8-7.4
        - Кластер 2 (аммоний): NH4=3.5-5.8, PO4=0.60-1.10, pH=6.3-6.9
        - Кластер 3 (промышленное): NH4=7.0-13.0, PO4=1.50-3.20, pH=5.4-6.3
        """
        
        if self.raster is None:
            raise ValueError("Сначала загрузите снимок методом load_raster()")
        
        # Границы снимка в проекционных координатах (Universal Transverse Mercator - UTM)
        left, bottom, right, top = self.raster.bounds
        print(f"Границы снимка: X: {left:.1f}-{right:.1f}, Y:{bottom:.1f}-{top:.1f}")
        
        # Генерируем случайные координаты внутри границ снимка
        np.random.seed(42) # Для воспроизводимости
        group_size = n_points // 4
        locations = [
            'Область 1',
            'Область 2',
            'Область 3',
            'Область 4',
            'Область 5' 
            ]

        """
        Задаем 4 четких кластера - для учебных целей (гауссовы
        распределения - сфера):
        увидеть работу алгоритма, поэкспериментировать с параметрами,
        получить красивую карту с цветными точками
        """
        cluster_params = [
            {'NH4': (0.05, 0.25), 'PO4': (0.02, 0.10), 'pH': (7.8, 8.3)}, # Чистая вода
            {'NH4': (1.2, 2.2), 'PO4': (0.25, 0.55), 'pH': (6.8, 7.4)}, # Органика
            {'NH4': (3.5, 5.8), 'PO4': (0.60, 1.10), 'pH': (6.3, 6.9)}, # Аммоний
            {'NH4': (7.0, 13.0), 'PO4': (1.50, 3.20), 'pH': (5.4, 6.3)} # Промышленность
        ]

        all_x, all_y, all_nh4, all_po4, all_ph, all_loc = [], [], [], [], [], []

        for i, params in enumerate(cluster_params):
            n_in_group = group_size if i < 3 else n_points - 3 * group_size

            # Разделенные координаты по кластерам
            x_start = left + i * (right - left) / 4.5
            x_end = left + (i + 1) * (right - left) / 4.5
            y_start = bottom + i * (top - bottom) / 4.5
            y_end = bottom + (i + 1) * (top - bottom) / 4.5
            x_group = np.random.uniform(x_start, x_end, n_in_group)
            y_group = np.random.uniform(y_start, y_end, n_in_group)

            # Параметры кластера
            nh4_group = np.random.uniform(params['NH4'][0], params['NH4'][1], n_in_group)
            po4_group = np.random.uniform(params['PO4'][0], params['PO4'][1], n_in_group)
            ph_group = np.random.uniform(params['pH'][0], params['pH'][1], n_in_group)
            loc_group = np.random.choice(locations, n_in_group)

            # Добавляем в списки
            all_x.extend(x_group)
            all_y.extend(y_group)
            all_nh4.extend(nh4_group)
            all_po4.extend(po4_group)
            all_ph.extend(ph_group)
            all_loc.extend(loc_group)

        # Создаем GeoDataFrame в CRS снимка
        self.points_gdf = gpd.GeoDataFrame({
            'id': range(1, n_points + 1),
            'NH4_mgL': np.round(all_nh4, 3),
            'PO4_mgL': np.round(all_po4, 3),
            'pH': np.round(all_ph, 2),
            'location': all_loc
        }, geometry=gpd.points_from_xy(all_x, all_y),
        crs=self.raster.crs)

        # В качестве примера финальная проверка для NH4
        print(f"Создано {len(self.points_gdf)} точек с 4 кластерами:")
        print(f"NH4: {self.points_gdf['NH4_mgL'].min():.2f} - {self.points_gdf['NH4_mgL'].max():.2f}")
        print(f"Группы по NH4: {self.points_gdf['NH4_mgL'].round(1).value_counts().sort_index().values}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.points_gdf.to_file(output_file, driver='GeoJSON')
        print(f"Сохранено в файле: {output_file}")
        
        return self.points_gdf

    def calculate_ndvi(self, red_idx=0, nir_idx=3):
        """Вычисление NDVI"""

        if self.bands is None:
            raise ValueError("Сначала загрузите снимок методомload_raster()")
        
        red = self.bands[red_idx].astype('float32')
        nir = self.bands[nir_idx].astype('float32')
        ndvi = (nir - red) / (nir + red + 1e-10)
        self.indices['NDVI'] = ndvi
        return ndvi

    
    def calculate_ndwi(self, green_idx=1, nir_idx=3):
        """Вычисление NDWI"""
        
        if self.bands is None:
            raise ValueError("Сначала загрузите снимок методом load_raster()")
        
        green = self.bands[green_idx].astype('float32')
        nir = self.bands[nir_idx].astype('float32')
        ndwi = (green - nir) / (green + nir + 1e-10)
        self.indices['NDWI'] = ndwi
        return ndwi

    
    def calculate_ndbi(self, swir_idx=4, nir_idx=3):
        """Вычисление NDBI"""

        if self.bands is None:
            raise ValueError("Сначала загрузите снимок методом load_raster()")
        
        swir = self.bands[swir_idx].astype('float32')
        nir = self.bands[nir_idx].astype('float32')
        ndbi = (swir - nir) / (swir + nir + 1e-10)
        self.indices['NDBI'] = ndbi
        return ndbi

    
    def calculate_all_indices(self):
        """Вычисление всех значений индексов"""

        self.calculate_ndvi()
        self.calculate_ndwi()
        self.calculate_ndbi()
        return self.indices

    def normalize(self, band):
        low, high = np.percentile(band, (2, 98))
        if high <= low:
            return np.zeros_like(band)
        return np.clip((band - low) / (high - low), 0, 1)

    def plot_with_points(self):
        """Все графики по очереди, каждый в отдельном окне"""

        if not self.indices or self.points_gdf is None:
            raise ValueError("Сначала рассчитайте индексы и создайте точки проб")

        self.plot_rgb_with_points()
        self.plot_index('NDVI', 'RdYlGn', -0.2, 0.8, 'NDVI (растительность)')
        self.plot_index('NDWI', 'Blues', -0.5, 0.5, 'NDWI (водные объекты)')
        self.plot_index('NDBI', 'Reds', -0.5, 0.5, 'NDBI (застройка)')
        self.plot_nh4_histogram()
        self.plot_locations()

    def plot_rgb_with_points(self, filename='rgb_points.png'):
        """RGB-снимок с точками проб, цвет точки зависит от NH4"""

        if self.raster is None or self.bands is None or self.points_gdf is None:
            raise ValueError("Сначала загрузите снимок и создайте точки проб")

        r, g, b = self.bands[0], self.bands[1], self.bands[2]
        rgb = np.dstack((self.normalize(r), self.normalize(g), self.normalize(b)))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)

        points_in_image = 0
        for idx, row in self.points_gdf.iterrows():
            row_idx, col = self.raster.index(row.geometry.x, row.geometry.y)
            if 0 <= col < rgb.shape[1] and 0 <= row_idx < rgb.shape[0]:
                points_in_image += 1
                color = 'red' if row['NH4_mgL'] > 1.0 else 'orange' if row['NH4_mgL'] > 0.5 else 'green'
                ax.scatter(col, row_idx, c=color, s=50, alpha=0.6)

        ax.axis('off')
        ax.set_title(f'Точки проб ({points_in_image})')

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_index(self, name, cmap, vmin, vmax, title, filename=None):
        """Карта одного спектрального индекса"""

        if name not in self.indices:
            raise ValueError(f"Индекс {name} не рассчитан")

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.indices[name], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.axis('off')

        filename = filename or f'{name.lower()}.png'
        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_nh4_histogram(self, filename='nh4_hist.png'):
        """Гистограмма концентрации NH4"""

        if self.points_gdf is None:
            raise ValueError("Сначала создайте точки проб")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.points_gdf['NH4_mgL'], bins=8, alpha=0.7, color='steelblue')
        ax.axvline(x=self.points_gdf['NH4_mgL'].mean(), color='red', linestyle='--', label='Среднее')
        ax.set_title('Распределение NH4')
        ax.set_xlabel('NH4, мг/л')
        ax.set_ylabel('Количество проб')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()


    def plot_locations(self, filename='locations.png'):
        """Распределение проб по локациям"""

        if self.points_gdf is None:
            raise ValueError("Сначала создайте точки проб")

        loc_counts = self.points_gdf['location'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(loc_counts.index.astype(str).tolist(), loc_counts.to_numpy(), alpha=0.7, color='steelblue')
        ax.set_title('Распределение проб по локациям')
        ax.set_ylabel('Количество проб')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()

    def find_optimal_eps_auto(self, X_scaled, min_samples=4):
        """Вычисление оптимального eps"""

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        distances, _ = neighbors.fit(X_scaled).kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples - 1])
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(distances)
        plt.title(f'k-distance график (k={min_samples})')
        plt.xlabel('Точки (отсортированы)')
        plt.ylabel(f'{min_samples}-е расстояние')
        plt.grid(True, alpha=0.3)
        
        try:
            kneedle = KneeLocator(range(len(distances)), distances, curve='convex',
                                    direction='increasing', S=1.0)
            if kneedle.knee is not None and kneedle.knee < len(distances) * 0.8:
                auto_eps = distances[kneedle.knee]
            else:
                auto_eps = np.percentile(distances, 40)
        except (ImportError, Exception):
            print("KneeLocator недоступен, используем перцентиль")
            auto_eps = np.percentile(distances, 40)
        
        best_eps = auto_eps
        best_score = -1
        eps_candidates = np.linspace(auto_eps * 0.6, auto_eps * 1.8, 7)

        plt.subplot(1, 2, 2)
        silhouette_scores = []

        for eps_test in eps_candidates:
            try:
                dbscan_test = DBSCAN(eps=eps_test, min_samples=min_samples)
                labels_test = dbscan_test.fit_predict(X_scaled)
                n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
                
                if n_clusters_test >= 2:
                    score = silhouette_score(X_scaled, labels_test)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
            except:
                silhouette_scores.append(-1)
    
        valid_indices = [i for i, s in enumerate(silhouette_scores) if s > -1]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i:
            silhouette_scores[i])
            best_eps = eps_candidates[best_idx]
            best_score = silhouette_scores[best_idx]
        
        print("\nМассив значений eps:")
        print(eps_candidates)
        print("Массив silhouette scores:")
        print(np.array(silhouette_scores))
        
        for i, eps_test in enumerate(eps_candidates):
            plt.scatter(eps_test, silhouette_scores[i], 
                        s=120, c='green' if i == np.argmax([s for s in silhouette_scores if s > -1]) else 'blue', 
                        label=f'eps={eps_test:.3f}' if i < 3 else "")
        plt.axvline(best_eps, color='red', linestyle='--', linewidth=2, label=f'Лучший: {best_eps:.3f} (score={best_score:.3f})')
        plt.xlabel('eps')
        plt.ylabel('Silhouette Score')
        plt.title('Автоподбор eps')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir + 'clustering', dpi=150)
        plt.show()

        print(f"Найден eps: {best_eps:.3f}")
        print(f"Silhouette: {best_score:.3f}")
        print(f"Диапазон расстояний: {distances[5]:.3f} - {np.percentile(distances, 75):.3f}")
        
        return best_eps

    def most_polluted_cluster(self, points_gdf, clusters):
        """Кластер с наибольшей средней концентрацией NH4 (без шума)"""

        nh4 = pd.Series(points_gdf['NH4_mgL'].to_numpy(), index=np.asarray(clusters))
        nh4 = nh4[nh4.index != -1]
        if nh4.empty:
            return None
        
        return nh4.groupby(level=0).mean().idxmax()

    def ndvi_near_water(self, x, y, radius_m=100):
        """NDVI в точке и наличие воды (NDWI > 0) в радиусе"""

        if self.raster is None or 'NDVI' not in self.indices or 'NDWI' not in self.indices:
            return None, False

        row, col = self.raster.index(x, y)
        height, width = self.indices['NDVI'].shape
        if not (0 <= row < height and 0 <= col < width):
            return None, False

        r = max(1, int(radius_m / self.raster.res[0]))
        window = self.indices['NDWI'][max(0, row - r):row + r + 1, max(0, col - r):col + r + 1]

        return float(self.indices['NDVI'][row, col]), bool((window > 0).any())


    def get_recommendations(self, cluster, high_cluster, x, y):
        """Рекомендации для всплывающего окна точки"""

        recommendations = []
        if high_cluster is not None and cluster == high_cluster:
            recommendations.append('Рекомендуется отбор повторной пробы')
        
        if cluster == -1:
            recommendations.append('Проверить координаты: возможна ошибка GPS')
        
        ndvi, near_water = self.ndvi_near_water(x, y)
        if ndvi is not None and ndvi < 0.2 and near_water:
            recommendations.append('Возможен сброс сточных вод')

        return recommendations

    def visualize_clusters_map(self, points_gdf=None, clusters=None, filename='clusters_map.html'):
        """Визуализация карты с кластерами проб"""

        if points_gdf is None:
            points_gdf = self.points_gdf
        if clusters is None:
            clusters = self.clusters
        
        if points_gdf is None or clusters is None:
            raise ValueError("Нет данных для визуализации кластеров")
        
        os.makedirs(os.path.dirname(filename) if
        os.path.dirname(filename) else '.', exist_ok=True)
        print("\nКонвертация координат для отображения на карте...")
        print(f"Исходная CRS точек: {getattr(points_gdf, 'crs', 'Не установлена')}")

        points_wgs84 = points_gdf.copy()

        current_crs = points_gdf.crs
        if current_crs and str(current_crs) != 'EPSG:4326':
            try:
                points_wgs84 = points_gdf.to_crs('EPSG:4326')
                print(f"Конвертировано в: {points_wgs84.crs}")
            except Exception as e:
                print(f"Ошибка конвертации: {e}")
                print("Пробуем принудительно установить CRS...")
                points_wgs84 = points_gdf.copy()
                points_wgs84.crs = 'EPSG:4326'
        
        print(f"\nДиапазон координат после конвертации:")
        print(f" Долгота (X): {points_wgs84.geometry.x.min():.6f} - {points_wgs84.geometry.x.max():.6f}")
        print(f" Широта (Y): {points_wgs84.geometry.y.min():.6f} - {points_wgs84.geometry.y.max():.6f}")
        
        center_lat = points_wgs84.geometry.y.mean()
        center_lon = points_wgs84.geometry.x.mean()
        
        print(f"Центр карты: {center_lat:.6f}, {center_lon:.6f}")
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'gray', 'brown', 'lightgray']

        high_cluster = self.most_polluted_cluster(points_gdf, clusters)

        points_added = 0
        for idx, row in points_wgs84.iterrows():
            cluster = clusters[idx]
            color = 'pink' if cluster == -1 else colors[cluster % len(colors)]

            radius = 5 + row['NH4_mgL'] * 3

            geom = points_gdf.loc[idx].geometry
            recommendations = self.get_recommendations(cluster, high_cluster, geom.x, geom.y)
            recommendations_html = ''.join(f'- {r}<br>' for r in recommendations)

            popup_text = f"""
                <b>Точка {row.get('id', idx)}</b><br>
                NH₄: {row['NH4_mgL']:.2f} мг/л<br>
                PO₄: {row['PO4_mgL']:.2f} мг/л<br>
                pH: {row['pH']:.2f}<br>
                Кластер: {cluster}<br>
                Локация: {row.get('location', 'Не указана')}<br>
                {'<b>Рекомендации:</b><br>' + recommendations_html if recommendations else ''}
            """

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
            points_added += 1

        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px;
            z-index: 1000; background-color: white; padding: 10px;
            border-radius: 5px; border: 2px solid grey; font-size:
            12px;">
                <b>Легенда</b><br>
                <span style="color: pink;">●</span> Шум (-1)<br>
                <span style="color: blue;">●</span> Кластер 0<br>
                <span style="color: red;">●</span> Кластер 1<br>
                <span style="color: green;">●</span> Кластер 2<br>
                <i>Размер = f(NH₄)</i>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник'
        ).add_to(m)
        folium.LayerControl().add_to(m)

        m.save(filename)
        print(f"\nКарта кластеров сохранена: {filename}")
        print(f"Добавлено точек на карту: {points_added} из {len(points_wgs84)}")
        
        webbrowser.open('file://' + os.path.realpath(filename))
        
        return m

    def plot_stl_component(self, series, title, ylabel, filename):
        """Одна компонента STL-разложения: наблюдаемый ряд, тренд или сезонность"""

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(series)
        ax.set_title(title)
        ax.set_xlabel('Дата')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_stl_residuals(self, resid, lower, upper, anomalies_mask, filename='ndvi_stl_resid.png'):
        """Остатки STL с коридором ±2σ и найденными аномалиями"""

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(resid)
        ax.fill_between(resid.index, lower, upper, color='gray', alpha=0.2, label='Коридор ±2σ')
        ax.scatter(resid.index[anomalies_mask], resid[anomalies_mask],
                    color='red', s=40, label='Аномалии ±2σ', zorder=5)
        ax.set_title('Остатки STL-разложения')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Остатки')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()
    
    
    def plot_ndvi_with_anomalies(self, ndvi, anomalies_mask, filename='ndvi_anomalies.png'):
        """Исходный ряд NDVI с отмеченными аномалиями"""

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(ndvi, label='NDVI')
        ax.scatter(ndvi.index[anomalies_mask], ndvi[anomalies_mask],
                    color='red', s=60, label='Обнаруженные аномалии', zorder=5)
        ax.set_title('NDVI с найденными аномалиями')
        ax.set_xlabel('Дата')
        ax.set_ylabel('NDVI')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()


    def plot_ndvi_histograms(self, ndvi_by_label, filename='ndvi_hist.png'):
        """Гистограммы NDVI нескольких снимков на одном графике"""

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green', 'red']
        for (label, ndvi), color in zip(ndvi_by_label.items(), colors):
            median = float(np.median(ndvi))
            ax.hist(ndvi.ravel(), bins=100, range=(-1, 1), density=True,
                    alpha=0.5, color=color, label=label)
            ax.axvline(median, color=color, linestyle='--', label=f'Медиана: {median:.2f}')
    
        ax.set_title('Распределение NDVI')
        ax.set_xlabel('NDVI')
        ax.set_ylabel('Плотность')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir + filename, dpi=150)
        plt.show()

    def analyze_ndvi_anomalies(self, n_years=10, seed=42):
        """
        Анализ NDVI: STL + поиск аномалий
        1. Создает временной ряд по паттернам MODIS леса
        2. Добавляет случайные аномалии (засухи/вырубки) - для
        иллюстрации
        3. STL-декомпозиция - остатки - выявление аномалий ±2σ
        """
        np.random.seed(seed)
        random.seed(seed)
        
        dates = pd.date_range(start='2015-01-01', periods=n_years * 12, freq='ME')
        months = dates.month
        seasonal_base = np.array([0.25, 0.28, 0.35, 0.45, 0.55, 0.70, 
                                    0.78, 0.75, 0.65, 0.50, 0.35, 0.28])
        seasonal = seasonal_base[months - 1]
        
        trend = np.linspace(0, -0.12, n_years * 12)
        
        noise = np.random.normal(0, 0.015, len(dates))

        anomalies = np.zeros(len(dates))
        anomaly_years = random.sample(range(2015, 2015 + n_years), min(2, n_years // 3))
        for year in anomaly_years:
            start_month = random.randint(4, 8)
            months_in_year = random.randint(4, 8)
            idx_start = (year - 2015) * 12 + start_month
            anomalies[idx_start:idx_start + months_in_year] -= np.random.uniform(0.15, 0.30)
        
        ndvi = np.clip(trend + seasonal + noise + anomalies, 0, 1)
        ts_df = pd.DataFrame({'date': dates, 'ndvi': ndvi}).set_index('date')
        ts_df.to_csv(os.path.join(self.output_dir, 'realistic_ndvi.csv'))
        
        print(f"Реалистичный NDVI {ts_df.index.min().year}-{ts_df.index.max().year}")
        print(f"Случайные аномалии добавлены в годы: {anomaly_years}")
        
        stl = STL(ts_df.ndvi, period=12, robust=True)
        result = stl.fit()
        
        resid_mean, resid_std = result.resid.mean(), result.resid.std()
        upper, lower = resid_mean + 2 * resid_std, resid_mean - 2 * resid_std

        anomalies_mask = (result.resid > upper) | (result.resid < lower)

        self.plot_stl_component(result.observed, 'Наблюдаемый NDVI', 'NDVI', 'ndvi_stl_observed.png')
        self.plot_stl_component(result.trend, 'Тренд NDVI', 'Тренд', 'ndvi_stl_trend.png')
        self.plot_stl_component(result.seasonal, 'Сезонность NDVI', 'Сезонность', 'ndvi_stl_seasonal.png')
        self.plot_stl_residuals(result.resid, lower, upper, anomalies_mask)
        self.plot_ndvi_with_anomalies(ts_df.ndvi, anomalies_mask)

        print("\nАномалии обнаружены автоматически")
        print("=" * 70)
        anomaly_df = ts_df[anomalies_mask].copy()
        anomaly_df['resid'] = result.resid[anomalies_mask]
        anomaly_df['zscore'] = (anomaly_df.resid - resid_mean) / resid_std
        print(anomaly_df.round(3))
        
        print("\nИнтерпретация:")
        for idx in anomaly_df.index:
            z = anomaly_df.loc[idx, 'zscore']
            year = idx.year
            if z < -2.5:
                event = "Сильная засуха/ вырубка"
            elif z < -1.8:
                event = "Умеренная засуха/болезнь"
            else:
                event = "Аномалия (анализ спутников/полей)"
            print(f" {idx.strftime('%Y-%m')}: z={z:.2f} → {event}")

        self.results['analyze_ndvi_anomalies'] = {
            'df': ts_df, 'stl': result, 'anomalies_mask':
            anomalies_mask,
            'anomaly_years_true': anomaly_years,
            'anomaly_years_detected': anomaly_df.index.year.unique()
        }
        return ts_df, anomaly_df