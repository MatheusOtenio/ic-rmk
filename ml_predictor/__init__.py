# ml_predictor - Sistema unificado de predição de evasão universitária

from .predictor import StudentDropoutPredictor
from .processor import DataProcessor
from .utils import setup_logging, create_directory_structure, validate_dataset
from .config import Config

__version__ = '1.0.0'
__author__ = 'IC Evasão Escolar'

# Exporta as classes e funções principais
__all__ = [
    'StudentDropoutPredictor',
    'DataProcessor',
    'Config',
    'setup_logging',
    'create_directory_structure',
    'validate_dataset'
]