"""
Абстрактний базовий клас для всіх стратегій.
Кожна конкретна стратегія повинна наслідувати Strategy і реалізувати generate_signals.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Базовий клас стратегії торгівлі.
    
    Всяка конкретна стратегія повинна:
    1. Наслідувати цей клас
    2. Реалізувати метод generate_signals(df)
    3. Повертати DataFrame з колонкою 'signal' (1=buy, -1=sell, 0=hold)
    """
    
    def __init__(self, name: str):
        """
        Ініціалізує стратегію.
        
        Args:
            name: Ім'я стратегії для логування
        """
        self.name = name
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерує сигнали купівлі/продажу на основі историчних даних.
        
        Args:
            df: DataFrame з OHLCV + індикаторами
               Обов'язкові колонки: open, high, low, close, volume
            
        Returns:
            Той же DataFrame з доданою колонкою 'signal':
            - 1: сигнал до купівлі
            - -1: сигнал до продажу
            - 0: без сигналу
        """
        pass
    
    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
