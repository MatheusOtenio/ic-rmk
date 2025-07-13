import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from .config import Config

# Configuração de logging
def setup_logging():
    """Configura o sistema de logging"""
    # Cria o diretório de logs se não existir
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Configura o logger
    logger = logging.getLogger('ml_predictor')
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Handler para arquivo
    file_handler = logging.FileHandler(f'{Config.LOGS_DIR}/ml_predictor.log')
    file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # Adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Criação da estrutura de diretórios
def create_directory_structure():
    """Cria a estrutura de diretórios para resultados"""
    structure = Config.get_output_structure()
    
    def create_dirs(structure, parent_path=''):
        for dir_name, subdirs in structure.items():
            path = os.path.join(parent_path, dir_name)
            os.makedirs(path, exist_ok=True)
            if subdirs:
                create_dirs(subdirs, path)
    
    create_dirs(structure)

# Validação de dados
def validate_dataset(data, dataset_name, logger):
    """Valida o dataset quanto a requisitos mínimos"""
    # Verifica colunas obrigatórias
    for column in Config.REQUIRED_COLUMNS:
        if column not in data.columns:
            error_msg = f"Coluna '{column}' não encontrada no dataset {dataset_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Verifica balanceamento de classes
    if 'target' in data.columns:
        class_counts = data['target'].value_counts()
        total_samples = len(data)
        
        # Verifica se há pelo menos duas classes
        if len(class_counts) < 2:
            error_msg = f"Dataset {dataset_name} possui apenas uma classe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verifica balanceamento mínimo
        min_class_ratio = class_counts.min() / total_samples
        if min_class_ratio < Config.MIN_CLASS_BALANCE:
            logger.warning(f"Dataset {dataset_name} possui desbalanceamento severo: {min_class_ratio:.2%} da classe minoritária")
        
        # Verifica amostras mínimas por classe
        if class_counts.min() < Config.MIN_SAMPLES_PER_CLASS:
            error_msg = f"Dataset {dataset_name} possui menos de {Config.MIN_SAMPLES_PER_CLASS} amostras na classe minoritária"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    return True

# Pré-processamento
def pre_process(data, dataset_name, logger):
    """Realiza o pré-processamento dos dados"""
    logger.info(f"Iniciando pré-processamento do dataset {dataset_name}")
    
    # Remove colunas Unnamed
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # Remove colunas com muitos valores ausentes
    missing_ratio = data.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > Config.MAX_MISSING_RATIO].index
    if len(columns_to_drop) > 0:
        logger.info(f"Removendo {len(columns_to_drop)} colunas com mais de {Config.MAX_MISSING_RATIO:.0%} de valores ausentes")
        data = data.drop(columns=columns_to_drop)
    
    # Processa colunas categóricas e numéricas
    for column in data.columns:
        if column == 'target':
            continue
            
        if data[column].dtype == 'object':
            logger.debug(f"Convertendo coluna categórica: {column}")
            try:
                data.loc[:, column] = pd.Categorical(data[column])
                data.loc[:, column] = LabelEncoder().fit_transform(data[column])
            except Exception as e:
                logger.warning(f"Erro ao converter coluna {column}: {str(e)}")
        
        # Converte vírgulas para pontos e transforma em numérico
        try:
            data.loc[:, column] = data[column].astype(str).str.replace(',', '.', regex=True)
            data.loc[:, column] = pd.to_numeric(data[column], errors='coerce')
        except Exception as e:
            logger.warning(f"Erro ao converter valores numéricos na coluna {column}: {str(e)}")
    
    logger.info(f"Pré-processamento concluído: {data.shape[0]} amostras, {data.shape[1]} features")
    return data

# Remoção de constantes
def const_remove(data, threshold, logger):
    """Remove features com variância abaixo do threshold"""
    logger.info(f"Removendo features constantes com threshold {threshold}")
    
    try:
        constant_filter = VarianceThreshold(threshold)
        constant_filter.fit(data)
        
        # Obtém as colunas que serão mantidas
        features_to_keep = data.columns[constant_filter.get_support()]
        features_removed = set(data.columns) - set(features_to_keep)
        
        if features_removed:
            logger.info(f"Features removidas por baixa variância: {features_removed}")
        
        data = data.iloc[:, constant_filter.get_support()]
        logger.info(f"Após remoção de constantes: {data.shape[1]} features")
        
        return data
    except Exception as e:
        logger.error(f"Erro na remoção de constantes: {str(e)}")
        raise

# Imputação
def imputation(data, logger):
    """Realiza imputação de valores ausentes"""
    logger.info(f"Realizando imputação de valores ausentes com estratégia '{Config.IMPUTATION_STRATEGY}'")
    
    try:
        # Verifica se há valores ausentes
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Imputando {missing_count} valores ausentes")
            
            imputer = SimpleImputer(strategy=Config.IMPUTATION_STRATEGY)
            df_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data=df_imputed, columns=data.columns)
        else:
            logger.info("Não há valores ausentes para imputar")
            
        return data
    except Exception as e:
        logger.error(f"Erro na imputação: {str(e)}")
        raise

# Remoção por correlação
def correlation_removal(data, threshold, logger):
    """Remove features altamente correlacionadas"""
    logger.info(f"Removendo features correlacionadas com threshold {threshold}")
    
    try:
        # Calcula a matriz de correlação
        corr = data.corr()
        
        # Identifica pares de features com alta correlação
        mask = (corr > threshold) | (corr < (-threshold))
        np.fill_diagonal(mask.values, False)  # Não considerar a diagonal
        
        # Identifica colunas para remover
        columns_to_drop = []
        for col in mask.columns:
            correlated_cols = mask.index[mask[col]].tolist()
            if correlated_cols:
                # Para cada par correlacionado, mantém a primeira e remove as outras
                columns_to_drop.extend(correlated_cols)
                # Marca as colunas já processadas para não repetir
                mask.loc[correlated_cols, :] = False
                mask.loc[:, correlated_cols] = False
        
        # Remove colunas duplicadas da lista
        columns_to_drop = list(set(columns_to_drop))
        
        if columns_to_drop:
            logger.info(f"Removendo {len(columns_to_drop)} features por alta correlação: {columns_to_drop}")
            data = data.drop(columns=columns_to_drop)
        else:
            logger.info("Nenhuma feature removida por correlação")
        
        return data
    except Exception as e:
        logger.error(f"Erro na remoção por correlação: {str(e)}")
        raise

# Visualização da distribuição de classes
def plot_class_distribution(data, dataset_name, output_dir=None):
    """Plota a distribuição de classes do dataset"""
    if output_dir is None:
        output_dir = Config.CLASS_DISTRIBUTION_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    if 'target' not in data.columns:
        return
    
    # Conta as classes
    class_counts = data['target'].value_counts()
    
    plt.figure(figsize=(10, 6))
    ax = class_counts.plot(kind='bar', color='black')
    plt.title(f'Distribuição de classes - {dataset_name}')
    plt.xlabel('Situação')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Adiciona rótulos com percentuais
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0f} ({height / class_counts.sum() * 100:.1f}%)', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='center', fontsize=10, color='gray', 
                   xytext=(0, 10), textcoords='offset points')
    
    # Salva o gráfico
    plt.savefig(f'{output_dir}/class-distribution-{dataset_name}.pdf', format='pdf')
    plt.close()

# Visualização da matriz de confusão
def plot_confusion_matrix(cm, dataset_name, algorithm, corr_threshold, const_threshold, seed, output_dir=None):
    """Plota a matriz de confusão"""
    if output_dir is None:
        output_dir = Config.CONFUSION_MATRIX_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verifica se a matriz tem o formato esperado
    if cm.shape != (2, 2):
        return
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel('Predições')
    plt.ylabel('Valores Reais')
    plt.title('Matriz de Confusão')
    
    # Salva o gráfico
    filename = f'cm-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}.pdf'
    plt.savefig(f'{output_dir}/{filename}', format='pdf')
    plt.close()