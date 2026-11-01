"""
Єдина точка налаштування логування для всього проекту.

Використання в будь-якому модулі:
    from bot.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """
    Повертає налаштований логер з виводом у консоль і (опціонально) у файл.

    :param name: ім'я логера, зазвичай __name__ модуля, що його викликає
    :param log_dir: каталог для файлу логів; якщо None — лог пишеться лише в консоль
    :return: готовий до використання logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Логер уже налаштований раніше (наприклад, при повторному імпорті модуля) —
        # виходимо одразу, щоб не додавати дублікати handler'ів і не дублювати рядки логів
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Ім'я файлу логів беремо з останнього сегмента імені логера (напр. "downloader.log")
        file_name = f"{name.split('.')[-1]}.log"
        file_handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Вимикаємо propagate, щоб повідомлення не дублювалися через кореневий логер
    logger.propagate = False
    return logger