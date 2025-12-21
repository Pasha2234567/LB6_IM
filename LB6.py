#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль прогнозирования временных рядов с адаптивным сглаживанием
"""

import csv
import numpy as np
from dataclasses import dataclass
from typing import Generator, Dict, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SmoothingMethod(Enum):
    """Методы адаптивного сглаживания"""
    EXPONENTIAL = "exponential"
    HOLT_LINEAR = "holt_linear"
    WINTERS_SEASONAL = "winters_seasonal"


@dataclass
class ModelCoefficients:
    """Коэффициенты модели прогнозирования"""
    baseline: float
    slope: float
    seasonal_factors: list
    smoothing_factors: tuple


class TimeSeriesProcessor:
    """Обработчик временных последовательностей"""
    
    def __init__(self, delimiter: str = ';'):
        self.delimiter = delimiter
        
    def extract_numeric_sequence(self, filepath: str) -> list:
        """Извлечение числовой последовательности из CSV"""
        numeric_values = []
        
        with open(filepath, 'r', encoding='utf-8') as datafile:
            csv_reader = csv.reader(datafile, delimiter=self.delimiter)
            
            for row in csv_reader:
                if len(row) > 1:
                    raw_value = row[1].strip()
                    if raw_value:
                        try:
                            # Нормализация числового формата
                            normalized = raw_value.replace(',', '.')
                            value = float(normalized)
                            numeric_values.append(value)
                        except (ValueError, TypeError):
                            continue
        return numeric_values


class RecursiveEstimator:
    """Рекурсивный оценщик параметров модели"""
    
    @staticmethod
    def estimate_initial_parameters(sequence: list, period: int) -> tuple:
        """Оценка начальных параметров через линейную регрессию"""
        if len(sequence) < period:
            raise ValueError(f"Необходимо минимум {period} наблюдений")
        
        # Выборка для инициализации
        init_data = sequence[:period]
        
        # Векторы для регрессии
        X = np.arange(period, dtype=float)
        Y = np.array(init_data, dtype=float)
        
        # Центрирование
        X_mean = X.mean()
        Y_mean = Y.mean()
        
        # Вычисление коэффициентов
        covariance = ((X - X_mean) * (Y - Y_mean)).sum()
        X_variance = ((X - X_mean) ** 2).sum()
        
        beta = covariance / X_variance if X_variance != 0 else 0.0
        alpha = Y_mean - beta * X_mean
        
        # Инициализация сезонных компонент
        seasonal = []
        for i in range(period):
            predicted = alpha + beta * i
            seasonal.append(Y[i] - predicted)
            
        return alpha, beta, seasonal
    
    @staticmethod
    def update_parameters_recursively(
        data_stream: list,
        period: int,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3
    ) -> ModelCoefficients:
        """Рекурсивное обновление параметров модели"""
        
        # Инициализация
        L, T, S = RecursiveEstimator.estimate_initial_parameters(data_stream, period)
        
        # Рекурсивная обработка
        for idx in range(period, len(data_stream)):
            seasonal_idx = idx % period
            prev_L = L
            
            # Коррекция уровня с учетом сезонности
            deseasonalized = data_stream[idx] - S[seasonal_idx]
            L = alpha * deseasonalized + (1 - alpha) * (L + T)
            
            # Коррекция тренда
            T = beta * (L - prev_L) + (1 - beta) * T
            
            # Коррекция сезонного фактора
            S[seasonal_idx] = gamma * (data_stream[idx] - L) + (1 - gamma) * S[seasonal_idx]
        
        return ModelCoefficients(
            baseline=L,
            slope=T,
            seasonal_factors=S,
            smoothing_factors=(alpha, beta, gamma)
        )


class ForecastingEngine:
    """Движок прогнозирования"""
    
    def __init__(self, model_params: ModelCoefficients):
        self.params = model_params
        
    def generate_horizon_predictions(self, steps: int) -> Generator[float, None, None]:
        """Генератор прогнозных значений"""
        L = self.params.baseline
        T = self.params.slope
        S = self.params.seasonal_factors
        period = len(S)
        
        for h in range(1, steps + 1):
            seasonal_idx = (h - 1) % period
            forecast_value = L + T * h + S[seasonal_idx]
            yield forecast_value
    
    def calculate_confidence_interval(self, predictions: list, confidence: float = 0.95) -> tuple:
        """Расчет доверительного интервала"""
        if not predictions:
            return 0.0, 0.0
        
        mean_val = np.mean(predictions)
        std_val = np.std(predictions)
        
        # Z-критерий для 95% доверительного уровня
        z_score = 1.96
        
        margin = z_score * std_val / np.sqrt(len(predictions))
        lower_bound = mean_val - margin
        upper_bound = mean_val + margin
        
        return lower_bound, upper_bound


class ResultFormatter:
    """Форматировщик результатов"""
    
    @staticmethod
    def create_report(model: ModelCoefficients, forecasts: list, metadata: Dict[str, Any]) -> str:
        """Создание структурированного отчета"""
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("АНАЛИТИЧЕСКИЙ ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ")
        report_lines.append("=" * 70)
        
        report_lines.append(f"\nМЕТАДАННЫЕ АНАЛИЗА:")
        report_lines.append(f"  • Датасет: {metadata.get('dataset', 'N/A')}")
        report_lines.append(f"  • Наблюдений: {metadata.get('observations', 0)}")
        report_lines.append(f"  • Сезонный цикл: {metadata.get('period', 12)} периодов")
        
        report_lines.append(f"\nОЦЕНКИ ПАРАМЕТРОВ МОДЕЛИ:")
        report_lines.append(f"  ┌─────────────────────────────┬─────────────────┐")
        report_lines.append(f"  │ Параметр                   │ Значение        │")
        report_lines.append(f"  ├─────────────────────────────┼─────────────────┤")
        report_lines.append(f"  │ Базовый уровень (L)        │ {model.baseline:15.6f} │")
        report_lines.append(f"  │ Трендовый коэффициент (T)  │ {model.slope:15.6f} │")
        report_lines.append(f"  │ Сглаживание уровня (α)     │ {model.smoothing_factors[0]:15.4f} │")
        report_lines.append(f"  │ Сглаживание тренда (β)     │ {model.smoothing_factors[1]:15.4f} │")
        report_lines.append(f"  │ Сглаживание сезонности (γ) │ {model.smoothing_factors[2]:15.4f} │")
        report_lines.append(f"  └─────────────────────────────┴─────────────────┘")
        
        report_lines.append(f"\nСЕЗОННЫЕ КОЭФФИЦИЕНТЫ:")
        for i, factor in enumerate(model.seasonal_factors, 1):
            report_lines.append(f"  Период {i:2d}: {factor:+10.6f} "
                              f"({'выше' if factor > 0 else 'ниже'} среднего)")
        
        report_lines.append(f"\nПРОГНОЗ НА {len(forecasts)} ПЕРИОДОВ:")
        for i, value in enumerate(forecasts, 1):
            trend_indicator = "↑" if value > forecasts[i-2] else "↓" if i > 1 else "→"
            report_lines.append(f"  Прогноз {i:2d}: {value:10.4f} {trend_indicator}")
        
        # Расчет статистики
        if forecasts:
            avg_forecast = np.mean(forecasts)
            min_forecast = np.min(forecasts)
            max_forecast = np.max(forecasts)
            
            report_lines.append(f"\nСТАТИСТИКА ПРОГНОЗА:")
            report_lines.append(f"  Среднее значение: {avg_forecast:.4f}")
            report_lines.append(f"  Диапазон: {min_forecast:.4f} ... {max_forecast:.4f}")
            report_lines.append(f"  Амплитуда: {(max_forecast - min_forecast):.4f}")
        
        report_lines.append("\n" + "=" * 70)
        
        return "\n".join(report_lines)


def execute_forecasting_pipeline(
    data_source: str,
    seasonal_cycle: int = 12,
    forecast_steps: int = 12,
    smoothing_params: tuple = (0.4, 0.3, 0.3)
) -> None:
    """
    Основной конвейер прогнозирования
    
    Args:
        data_source: Путь к файлу с данными
        seasonal_cycle: Длина сезонного цикла
        forecast_steps: Горизонт прогнозирования
        smoothing_params: Коэффициенты сглаживания (α, β, γ)
    """
    
    # Этап 1: Загрузка данных
    print("🔍 Загрузка данных...")
    processor = TimeSeriesProcessor()
    try:
        time_series = processor.extract_numeric_sequence(data_source)
        if not time_series:
            print("Ошибка: данные не загружены")
            return
        print(f"✓ Загружено {len(time_series)} значений")
    except FileNotFoundError:
        print(f"Ошибка: файл '{data_source}' не найден")
        return
    
    # Этап 2: Проверка данных
    if len(time_series) < 2 * seasonal_cycle:
        print(f"Ошибка: недостаточно данных. Требуется минимум {2*seasonal_cycle} точек")
        return
    
    # Этап 3: Построение модели
    print("📈 Построение адаптивной модели...")
    estimator = RecursiveEstimator()
    
    alpha, beta, gamma = smoothing_params
    model = estimator.update_parameters_recursively(
        time_series,
        seasonal_cycle,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    print("✓ Модель успешно обучена")
    
    # Этап 4: Прогнозирование
    print("🔮 Генерация прогнозов...")
    engine = ForecastingEngine(model)
    predictions = list(engine.generate_horizon_predictions(forecast_steps))
    
    # Этап 5: Анализ результатов
    confidence_lower, confidence_upper = engine.calculate_confidence_interval(predictions)
    
    # Этап 6: Форматирование отчета
    metadata = {
        'dataset': data_source,
        'observations': len(time_series),
        'period': seasonal_cycle,
        'confidence': (confidence_lower, confidence_upper)
    }
    
    formatter = ResultFormatter()
    report = formatter.create_report(model, predictions, metadata)
    
    # Этап 7: Вывод результатов
    print("\n" + report)
    
    # Дополнительная информация
    print("\n📊 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
    print(f"   Доверительный интервал 95%: [{confidence_lower:.4f}, {confidence_upper:.4f}]")
    print(f"   Относительная ошибка прогноза: {np.std(predictions)/np.mean(predictions)*100:.2f}%")
    
    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    if abs(model.slope) > 0.1:
        trend_dir = "растущий" if model.slope > 0 else "падающий"
        print(f"   • Обнаружен {trend_dir} тренд ({model.slope:.4f} за период)")
    else:
        print("   • Тренд незначителен, стабильная ситуация")
    
    max_seasonal = max(model.seasonal_factors)
    min_seasonal = min(model.seasonal_factors)
    if max_seasonal - min_seasonal > 1.0:
        print("   • Выраженная сезонность: рекомендуется учет сезонных факторов")


def main():
    """Точка входа в приложение"""
    
    # Конфигурация
    CONFIG = {
        'input_file': 'LAB6.csv',
        'seasonal_period': 12,
        'prediction_horizon': 12,
        'smoothing_parameters': (0.4, 0.3, 0.3)  # α, β, γ
    }
    
    print("=" * 60)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ v3.0")
    print("Модель: адаптивное тройное экспоненциальное сглаживание")
    print("=" * 60)
    
    try:
        execute_forecasting_pipeline(
            data_source=CONFIG['input_file'],
            seasonal_cycle=CONFIG['seasonal_period'],
            forecast_steps=CONFIG['prediction_horizon'],
            smoothing_params=CONFIG['smoothing_parameters']
        )
    except KeyboardInterrupt:
        print("\n\nОперация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
