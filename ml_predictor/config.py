# Configuração centralizada para o sistema de predição de evasão universitária

class Config:
    # Parâmetros de experimentação
    CORRELATION_THRESHOLDS = [0.8, 0.85, 0.9, 0.95]
    CONSTANT_THRESHOLDS = [0.05, 0.1, 0.15, 0.2]
    SEEDS = [145, 278, 392, 49, 203, 411, 89, 356, 27, 489]
    ALGORITHMS = {
        'dt': 'Decision Tree',
        'rf': 'Random Forest',
        'neigh': 'K-Neighbors',
        'nb': 'Naive Bayes'
    }
    
    # Configurações de processamento
    IMPUTATION_STRATEGY = 'mean'
    MIN_CLASS_BALANCE = 0.1  # Proporção mínima da classe minoritária
    
    # Configurações de saída
    RESULTS_DIR = 'results'
    EXPERIMENTS_DIR = f'{RESULTS_DIR}/experiments'
    PREDICTIONS_DIR = f'{RESULTS_DIR}/predictions'
    PLOTS_DIR = f'{RESULTS_DIR}/plots'
    LOGS_DIR = f'{RESULTS_DIR}/logs'
    
    # Subdiretórios de plots
    CLASS_DISTRIBUTION_DIR = f'{PLOTS_DIR}/class-distribution'
    CONFUSION_MATRIX_DIR = f'{PLOTS_DIR}/confusion-matrix'
    FEATURE_IMPORTANCE_DIR = f'{PLOTS_DIR}/feature-importance'
    DECISION_TREE_DIR = f'{PLOTS_DIR}/decision-tree'
    
    # Configurações de validação
    REQUIRED_COLUMNS = ['target']
    MIN_SAMPLES_PER_CLASS = 10
    MAX_MISSING_RATIO = 0.5  # Máximo de valores ausentes permitidos por coluna
    
    # Configurações de logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @staticmethod
    def get_output_structure():
        """Retorna a estrutura de diretórios para saída de resultados"""
        return {
            Config.RESULTS_DIR: {
                Config.EXPERIMENTS_DIR: {},
                Config.PREDICTIONS_DIR: {},
                Config.PLOTS_DIR: {
                    Config.CLASS_DISTRIBUTION_DIR: {},
                    Config.CONFUSION_MATRIX_DIR: {},
                    Config.FEATURE_IMPORTANCE_DIR: {},
                    Config.DECISION_TREE_DIR: {}
                },
                Config.LOGS_DIR: {}
            }
        }