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
    IMPUTATION_STRATEGY = 'mean'  # Estratégias: 'mean', 'median', 'most_frequent', 'constant'
    IMPUTATION_CONSTANT = 0  # Valor constante para imputação quando IMPUTATION_STRATEGY='constant'
    MIN_CLASS_BALANCE = 0.1  # Proporção mínima da classe minoritária
    
    # Configurações de balanceamento de classes
    APPLY_CLASS_BALANCING = False  # Se True, aplica técnicas de balanceamento
    BALANCING_STRATEGY = 'smote'  # Estratégias: 'smote', 'random_over', 'random_under', 'adasyn'
    
    # Configurações de processamento de dados
    PRESERVE_CATEGORICAL_COLUMNS = True  # Se True, preserva colunas categóricas originais
    PROCESS_DATE_COLUMNS = False  # Se True, processa colunas de data (desativado por padrão para evitar erros)
    NORMALIZE_FEATURES = False  # Se True, normaliza features numéricas
    
    # Configurações de saída
    RESULTS_DIR = 'results'
    EXPERIMENTS_DIR = 'results/experiments'
    PREDICTIONS_DIR = 'results/predictions'
    PLOTS_DIR = 'results/plots'
    LOGS_DIR = 'results/logs'
    
    # Subdiretórios de plots
    CLASS_DISTRIBUTION_DIR = 'results/plots/class-distribution'
    CONFUSION_MATRIX_DIR = 'results/plots/confusion-matrix'
    FEATURE_IMPORTANCE_DIR = 'results/plots/feature-importance'
    DECISION_TREE_DIR = 'results/plots/decision-tree'
    
    # Configurações de visualização
    GENERATE_PLOTS = True  # Se False, não gera nenhuma visualização
    PLOT_FORMAT = 'png'  # Formato dos gráficos: 'pdf', 'png', 'svg'
    PLOT_DPI = 300  # Resolução dos gráficos (para formatos raster)
    
    # Controle de visualizações específicas
    PLOT_CLASS_DISTRIBUTION = True
    PLOT_CONFUSION_MATRIX = True
    PLOT_FEATURE_IMPORTANCE = True
    PLOT_DECISION_TREE = False  # Árvores de decisão podem ser muito grandes e consumir muito espaço
    
    # Configurações de consolidação de relatórios
    GENERATE_CONSOLIDATED_REPORT = True  # Se True, gera um relatório consolidado
    GENERATE_HTML_REPORT = True  # Se True, gera um relatório HTML
    MAX_PLOTS_PER_REPORT = 10  # Número máximo de plots por relatório
    
    # Configurações de controle de saída
    OUTPUT_FORMATS = {  # Formatos de saída habilitados
        'pdf': True,
        'csv': True,
        'json': True,
        'html': True
    }
    SAVE_PREDICTIONS = True  # Se True, salva as predições individuais
    SAVE_INDIVIDUAL_PLOTS = True  # Se False, gera apenas visualizações no relatório consolidado
    
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
            # Diretórios principais
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