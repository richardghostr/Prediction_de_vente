"""
Utilitaires de logging centralisés pour le projet.

Objectifs :
- Fournir une configuration homogène (format, niveau, handlers).
- Éviter la duplication de configuration dans chaque module.

Usage :
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

from __future__ import annotations

import logging
import os
from typing import Optional


_CONFIGURED = False


def _configure_root_logger(level: Optional[str] = None) -> None:
    """Configure le logger racine une seule fois."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Éviter de dupliquer les handlers si une autre config a déjà été appliquée
    if not root.handlers:
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Retourne un logger configuré pour le projet.

    Le premier appel configure le logger racine; les suivants récupèrent
    simplement un logger nommé via logging.getLogger(name).
    """
    _configure_root_logger(level=level)
    logger = logging.getLogger(name)
    return logger


__all__ = ["get_logger"]

