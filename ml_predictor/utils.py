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
            # Verifica se o diretório já é um caminho absoluto ou relativo completo
            if os.path.isabs(dir_name) or dir_name.startswith('results/'):
                path = dir_name
            else:
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
    
    # Identifica e converte colunas de data se configurado
    if Config.PROCESS_DATE_COLUMNS:
        date_columns = [col for col in data.columns if 'data' in col.lower() or 'date' in col.lower()]
        for col in date_columns:
            try:
                # Converte para datetime com formato brasileiro (dia/mês/ano)
                data[col] = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
                
                # Extrai características das datas
                if not data[col].isna().all():
                    # Cria novas colunas com as características extraídas
                    data[f'{col}_ano'] = data[col].dt.year
                    data[f'{col}_mes'] = data[col].dt.month
                    data[f'{col}_dia_semana'] = data[col].dt.dayofweek
                    
                    # Remove a coluna original de data para evitar problemas de conversão
                    data = data.drop(columns=[col])
                    logger.info(f"Coluna de data {col} processada, expandida e removida")
            except Exception as e:
                logger.warning(f"Erro ao processar coluna de data {col}: {str(e)}")
    
    # Remove colunas com muitos valores ausentes
    missing_ratio = data.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > Config.MAX_MISSING_RATIO].index
    if len(columns_to_drop) > 0:
        logger.info(f"Removendo {len(columns_to_drop)} colunas com mais de {Config.MAX_MISSING_RATIO:.0%} de valores ausentes")
        data = data.drop(columns=columns_to_drop)
    
    # Processa colunas categóricas e numéricas
    categorical_columns = []
    for column in data.columns:
        if column == 'target':
            continue
        
        # Tenta converter valores numéricos com vírgula para ponto
        if data[column].dtype == 'object':
            # Verifica se a coluna parece conter números com vírgula
            sample = data[column].dropna().astype(str).str.replace(',', '.', regex=True).head(100)
            numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
            
            # Se mais de 80% dos valores são numéricos após substituição, converte
            if numeric_count > 0.8 * len(sample) and len(sample) > 0:
                try:
                    logger.debug(f"Convertendo coluna numérica com vírgula: {column}")
                    data.loc[:, column] = data[column].astype(str).str.replace(',', '.', regex=True)
                    data.loc[:, column] = pd.to_numeric(data[column], errors='coerce')
                except Exception as e:
                    logger.warning(f"Erro ao converter valores numéricos na coluna {column}: {str(e)}")
                    categorical_columns.append(column)
            else:
                # Processa como categórica
                logger.debug(f"Convertendo coluna categórica: {column}")
                categorical_columns.append(column)
                try:
                    # Preserva a coluna original antes de codificar
                    if Config.PRESERVE_CATEGORICAL_COLUMNS:
                        data[f"{column}_original"] = data[column]
                    
                    # Codifica a coluna categórica
                    data.loc[:, column] = pd.Categorical(data[column])
                    data.loc[:, column] = LabelEncoder().fit_transform(data[column])
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna categórica {column}: {str(e)}")
                    # Se falhar na codificação, remove a coluna
                    if column in data.columns:
                        logger.warning(f"Removendo coluna problemática: {column}")
                        data = data.drop(columns=[column])
                        if f"{column}_original" in data.columns:
                            data = data.drop(columns=[f"{column}_original"])
        
        # Para colunas já numéricas, não precisa de conversão
        elif pd.api.types.is_numeric_dtype(data[column]):
            pass
    
    logger.info(f"Processadas {len(categorical_columns)} colunas categóricas")
    
    logger.info(f"Pré-processamento concluído: {data.shape[0]} amostras, {data.shape[1]} features")
    return data

# Remoção de constantes
def const_remove(data, threshold, logger):
    """Remove features com variância abaixo do threshold"""
    logger.info(f"Removendo features constantes com threshold {threshold}")
    
    try:
        # Verifica se há colunas não numéricas e as remove
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Removendo {len(non_numeric_cols)} colunas não numéricas antes da verificação de constantes: {list(non_numeric_cols)}")
            data = data.select_dtypes(include=['number'])
            
            # Se não restarem colunas, retorna um DataFrame vazio
            if data.shape[1] == 0:
                logger.error("Nenhuma coluna numérica disponível após filtragem")
                return pd.DataFrame()
        
        # Verifica se há valores NaN e os substitui pela média da coluna
        if data.isnull().any().any():
            logger.warning("Substituindo valores NaN pela média das colunas antes da verificação de constantes")
            data = data.fillna(data.mean())
        
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
        # Em vez de propagar o erro, retorna o DataFrame original
        logger.warning("Retornando dataset original devido a erro na remoção de constantes")
        return data

# Imputação
def imputation(data, logger):
    """Realiza imputação de valores ausentes"""
    logger.info(f"Realizando imputação de valores ausentes com estratégia '{Config.IMPUTATION_STRATEGY}'")
    
    try:
        # Verifica se há valores ausentes
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Imputando {missing_count} valores ausentes")
            
            # Configura o imputer de acordo com a estratégia
            if Config.IMPUTATION_STRATEGY == 'constant':
                imputer = SimpleImputer(strategy=Config.IMPUTATION_STRATEGY, fill_value=Config.IMPUTATION_CONSTANT)
            else:
                imputer = SimpleImputer(strategy=Config.IMPUTATION_STRATEGY)
            
            # Aplica imputação por tipo de coluna
            numeric_cols = data.select_dtypes(include=['number']).columns
            categorical_cols = data.select_dtypes(exclude=['number']).columns
            
            # Imputação para colunas numéricas
            if len(numeric_cols) > 0:
                data_numeric = data[numeric_cols]
                if data_numeric.isnull().sum().sum() > 0:
                    df_imputed_numeric = imputer.fit_transform(data_numeric)
                    data[numeric_cols] = df_imputed_numeric
            
            # Imputação para colunas categóricas (usando most_frequent)
            if len(categorical_cols) > 0:
                data_cat = data[categorical_cols]
                if data_cat.isnull().sum().sum() > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed_cat = cat_imputer.fit_transform(data_cat)
                    data[categorical_cols] = df_imputed_cat
        else:
            logger.info("Não há valores ausentes para imputar")
            
        return data
    except Exception as e:
        logger.error(f"Erro na imputação: {str(e)}")
        raise

# Balanceamento de classes
def balance_classes(X, y, logger):
    """Aplica técnicas de balanceamento de classes"""
    if not Config.APPLY_CLASS_BALANCING:
        return X, y
    
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        
        logger.info(f"Aplicando balanceamento de classes com estratégia '{Config.BALANCING_STRATEGY}'")
        
        # Verifica a distribuição original
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Distribuição original: {class_counts.to_dict()}")
        
        # Aplica a estratégia de balanceamento
        if Config.BALANCING_STRATEGY == 'smote':
            balancer = SMOTE(random_state=42)
        elif Config.BALANCING_STRATEGY == 'random_over':
            balancer = RandomOverSampler(random_state=42)
        elif Config.BALANCING_STRATEGY == 'random_under':
            balancer = RandomUnderSampler(random_state=42)
        elif Config.BALANCING_STRATEGY == 'adasyn':
            balancer = ADASYN(random_state=42)
        else:
            logger.warning(f"Estratégia de balanceamento '{Config.BALANCING_STRATEGY}' não reconhecida. Usando SMOTE.")
            balancer = SMOTE(random_state=42)
        
        # Aplica o balanceamento
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        # Verifica a nova distribuição
        new_class_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"Distribuição após balanceamento: {new_class_counts.to_dict()}")
        
        return X_resampled, y_resampled
    
    except ImportError:
        logger.warning("Biblioteca 'imbalanced-learn' não encontrada. Instale com 'pip install imbalanced-learn' para usar balanceamento de classes.")
        return X, y
    except Exception as e:
        logger.error(f"Erro no balanceamento de classes: {str(e)}")
        return X, y

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
        # Em vez de propagar o erro, retorna o DataFrame original
        logger.warning("Retornando dataset original devido a erro na remoção por correlação")
        return data

# Balanceamento de classes
def balance_classes(X, y, logger):
    """Realiza balanceamento de classes se configurado"""
    if not Config.APPLY_CLASS_BALANCING:
        logger.info("Balanceamento de classes não aplicado (desativado na configuração)")
        return X, y
    
    try:
        # Verifica a distribuição de classes atual
        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        minority_count = class_counts[minority_class]
        majority_class = np.argmax(class_counts)
        majority_count = class_counts[majority_class]
        
        # Calcula a proporção da classe minoritária
        minority_ratio = minority_count / len(y)
        
        logger.info(f"Distribuição original: Classe {minority_class}: {minority_count}, "
                   f"Classe {majority_class}: {majority_count}, "
                   f"Proporção minoritária: {minority_ratio:.2%}")
        
        # Verifica se o balanceamento é necessário
        if minority_ratio >= Config.MIN_CLASS_BALANCE:
            logger.info(f"Balanceamento não necessário. Proporção da classe minoritária ({minority_ratio:.2%}) "
                       f"é maior que o mínimo configurado ({Config.MIN_CLASS_BALANCE:.2%})")
            return X, y
        
        # Aplica a estratégia de balanceamento configurada
        logger.info(f"Aplicando balanceamento com estratégia '{Config.BALANCING_STRATEGY}'")
        
        if Config.BALANCING_STRATEGY == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42)
        elif Config.BALANCING_STRATEGY == 'random_over':
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
        elif Config.BALANCING_STRATEGY == 'random_under':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
        elif Config.BALANCING_STRATEGY == 'adasyn':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=42)
        else:
            logger.warning(f"Estratégia de balanceamento '{Config.BALANCING_STRATEGY}' não reconhecida. Usando SMOTE.")
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42)
        
        # Aplica o balanceamento
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Verifica a nova distribuição
        new_class_counts = np.bincount(y_resampled)
        logger.info(f"Distribuição após balanceamento: "
                   f"Classe {minority_class}: {new_class_counts[minority_class]}, "
                   f"Classe {majority_class}: {new_class_counts[majority_class]}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"Erro no balanceamento de classes: {str(e)}")
        logger.warning("Retornando dataset original devido a erro no balanceamento")
        return X, y

# Visualização da distribuição de classes
def plot_class_distribution(data, dataset_name, output_dir=None):
    """Plota a distribuição de classes do dataset"""
    # Verifica se a geração de plots está habilitada
    if not Config.GENERATE_PLOTS or not Config.PLOT_CLASS_DISTRIBUTION or not Config.SAVE_INDIVIDUAL_PLOTS:
        return
    
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
    
    # Salva o gráfico nos formatos configurados
    for fmt, enabled in Config.OUTPUT_FORMATS.items():
        if enabled and fmt in ['pdf', 'png', 'svg']:
            filename = f'class-distribution-{dataset_name}.{fmt}'
            if fmt in ['png', 'jpg']:
                plt.savefig(f'{output_dir}/{filename}', format=fmt, dpi=Config.PLOT_DPI)
            else:
                plt.savefig(f'{output_dir}/{filename}', format=fmt)
    
    plt.close()

# Visualização da matriz de confusão
def plot_confusion_matrix(cm, dataset_name, algorithm, corr_threshold, const_threshold, seed, output_dir=None):
    """Plota a matriz de confusão"""
    # Verifica se a geração de plots está habilitada
    if not Config.GENERATE_PLOTS or not Config.PLOT_CONFUSION_MATRIX or not Config.SAVE_INDIVIDUAL_PLOTS:
        return
    
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
    plt.title(f'Matriz de Confusão - {dataset_name} - {algorithm}')
    
    # Adiciona métricas derivadas da matriz de confusão
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}', 
                fontsize=9, ha='left')
    
    # Salva o gráfico nos formatos configurados
    base_filename = f'cm-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}'
    
    for fmt, enabled in Config.OUTPUT_FORMATS.items():
        if enabled and fmt in ['pdf', 'png', 'svg']:
            filename = f'{base_filename}.{fmt}'
            if fmt in ['png', 'jpg']:
                plt.savefig(f'{output_dir}/{filename}', format=fmt, dpi=Config.PLOT_DPI)
            else:
                plt.savefig(f'{output_dir}/{filename}', format=fmt)
    
    plt.close()