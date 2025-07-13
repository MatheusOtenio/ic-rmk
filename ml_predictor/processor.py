import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import logging

from .config import Config
from .utils import pre_process, const_remove, imputation, correlation_removal, plot_confusion_matrix

class DataProcessor:
    """Classe responsável pelo processamento de dados e execução de experimentos"""
    
    def __init__(self, logger=None):
        """Inicializa o processador de dados"""
        self.logger = logger or logging.getLogger('ml_predictor')
        self.models = self._load_models()
    
    def _load_models(self):
        """Carrega os modelos de machine learning"""
        self.logger.info("Carregando modelos de machine learning")
        
        models = {
            'dt': DecisionTreeClassifier(random_state=42),
            'rf': RandomForestClassifier(random_state=42),
            'neigh': KNeighborsClassifier(),
            'nb': GaussianNB()
        }
        
        return models
    
    def load_dataset(self, dataset_path):
        """Carrega um dataset a partir do caminho especificado"""
        try:
            self.logger.info(f"Carregando dataset: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                error_msg = f"Arquivo não encontrado: {dataset_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Carrega o dataset
            data = pd.read_csv(dataset_path)
            
            # Extrai o nome do dataset do caminho
            dataset_name = os.path.basename(dataset_path).replace('.csv', '')
            
            self.logger.info(f"Dataset carregado: {dataset_name} com {data.shape[0]} amostras e {data.shape[1]} colunas")
            return data, dataset_name
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar dataset {dataset_path}: {str(e)}")
            raise
    
    def train_test(self, X, y, algorithm, seed, n_splits=5):
        """Treina e testa um modelo usando validação cruzada"""
        try:
            self.logger.info(f"Iniciando treinamento e teste com algoritmo {algorithm}, seed {seed}")
            
            # Verifica se o algoritmo é válido
            if algorithm not in self.models:
                error_msg = f"Algoritmo inválido: {algorithm}. Opções disponíveis: {list(self.models.keys())}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Verifica se há amostras suficientes para o número de splits
            if len(y) < n_splits:
                self.logger.warning(f"Número de amostras ({len(y)}) menor que n_splits ({n_splits}). Ajustando n_splits para {len(y)}")
                n_splits = min(len(y), n_splits)
            
            # Obtém o classificador
            classifier = self.models[algorithm]
            if hasattr(classifier, 'random_state'):
                classifier.random_state = seed
            
            # Prepara a validação cruzada
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            
            # Métricas a serem calculadas
            scoring = {
                'accuracy': 'accuracy',
                'balanced_accuracy': 'balanced_accuracy',
                'f1': 'f1',
                'precision': 'precision',
                'recall': 'recall'
            }
            
            # Executa a validação cruzada
            try:
                cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, return_estimator=True)
                
                # Treina o modelo final com todos os dados para obter a matriz de confusão
                classifier.fit(X, y)
                y_pred = classifier.predict(X)
                cm = confusion_matrix(y, y_pred)
                
                # Calcula probabilidades se o modelo suportar
                probabilities = None
                if hasattr(classifier, 'predict_proba'):
                    probabilities = classifier.predict_proba(X)[:, 1]  # Probabilidade da classe positiva
                
                # Prepara o dicionário de resultados
                results = {
                    'accuracy': np.mean(cv_results['test_accuracy']),
                    'balanced_accuracy': np.mean(cv_results['test_balanced_accuracy']),
                    'f1_score': np.mean(cv_results['test_f1']),
                    'precision': np.mean(cv_results['test_precision']),
                    'recall': np.mean(cv_results['test_recall']),
                    'confusion_matrix': cm.tolist(),
                    'estimator': classifier,  # Modelo treinado
                    'true_labels': y.tolist(),
                    'predicted_labels': y_pred.tolist(),
                    'probabilities': probabilities.tolist() if probabilities is not None else None
                }
                
                # Informações sobre as classes
                class_counts = pd.Series(y).value_counts()
                dropout_count = class_counts.get(1, 0)
                regular_count = class_counts.get(0, 0)
                
                # Adiciona informações sobre as classes
                infos = {
                    'dropout': dropout_count,
                    'regular': regular_count,
                    'total': len(y),
                    'features': X.shape[1]
                }
                
                self.logger.info(f"Treinamento concluído com sucesso. Acurácia média: {results['accuracy']:.4f}")
                return results, infos
                
            except Exception as e:
                self.logger.error(f"Erro durante a validação cruzada: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Erro no train_test: {str(e)}")
            raise
    
    def execute(self, data, dataset_name, algorithm, corr_threshold, const_threshold, seed):
        """Executa um experimento completo"""
        try:
            self.logger.info(f"Executando experimento: {dataset_name}, {algorithm}, corr={corr_threshold}, const={const_threshold}, seed={seed}")
            
            # Verifica se a coluna target existe
            if 'target' not in data.columns:
                error_msg = f"Coluna 'target' não encontrada no dataset {dataset_name}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Separa features e target
            y = data['target']
            X = data.drop(columns=['target'])
            
            # Pré-processamento
            X = pre_process(X, dataset_name, self.logger)
            
            # Remoção de constantes
            X = const_remove(X, const_threshold, self.logger)
            
            # Imputação
            X = imputation(X, self.logger)
            
            # Remoção por correlação
            X = correlation_removal(X, corr_threshold, self.logger)
            
            # Verifica se restaram features após o processamento
            if X.shape[1] == 0:
                error_msg = f"Nenhuma feature restante após o processamento para {dataset_name}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Treina e testa o modelo
            results, infos = self.train_test(X, y, algorithm, seed)
            
            # Gera a matriz de confusão
            if 'confusion_matrix' in results:
                cm = np.array(results['confusion_matrix'])
                plot_confusion_matrix(cm, dataset_name, algorithm, corr_threshold, const_threshold, seed)
            
            # Gera visualizações específicas para cada algoritmo
            if algorithm == 'dt' and 'estimator' in results:
                self._plot_decision_tree(results['estimator'], X, dataset_name, corr_threshold, const_threshold, seed)
            
            if algorithm == 'rf' and 'estimator' in results:
                self._plot_feature_importance(results['estimator'], X, dataset_name, corr_threshold, const_threshold, seed)
            
            # Prepara o resultado final
            experiment_result = {
                'dataset_info': {
                    'name': dataset_name,
                    'samples': infos['total'],
                    'features': infos['features'],
                    'class_distribution': {
                        '0': infos['regular'],
                        '1': infos['dropout']
                    }
                },
                'experiment_config': {
                    'algorithm': Config.ALGORITHMS.get(algorithm, algorithm),
                    'corr_threshold': corr_threshold,
                    'const_threshold': const_threshold,
                    'seed': seed
                },
                'results': {
                    'accuracy': results['accuracy'],
                    'balanced_accuracy': results['balanced_accuracy'],
                    'f1_score': results['f1_score'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'confusion_matrix': results['confusion_matrix']
                },
                'predictions': {
                    'true_labels': results['true_labels'],
                    'predicted_labels': results['predicted_labels'],
                    'probabilities': results['probabilities']
                }
            }
            
            self.logger.info(f"Experimento concluído com sucesso: {dataset_name}, {algorithm}")
            return experiment_result
            
        except Exception as e:
            self.logger.error(f"Erro na execução do experimento: {str(e)}")
            raise
    
    def _plot_decision_tree(self, tree_model, X, dataset_name, corr_threshold, const_threshold, seed):
        """Plota a árvore de decisão"""
        try:
            output_dir = Config.DECISION_TREE_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(20, 10))
            plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=['0', '1'], rounded=True)
            plt.title(f'Árvore de Decisão - {dataset_name}')
            
            # Salva o gráfico
            filename = f'dt-{dataset_name}-{corr_threshold}-{const_threshold}-{seed}.pdf'
            plt.savefig(f'{output_dir}/{filename}', format='pdf')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Erro ao plotar árvore de decisão: {str(e)}")
    
    def _plot_feature_importance(self, rf_model, X, dataset_name, corr_threshold, const_threshold, seed):
        """Plota a importância das features para Random Forest"""
        try:
            output_dir = Config.FEATURE_IMPORTANCE_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            # Obtém a importância das features
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Limita a 20 features mais importantes
            n_features = min(20, X.shape[1])
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Importância das Features - {dataset_name}')
            plt.bar(range(n_features), importances[indices[:n_features]], align='center')
            plt.xticks(range(n_features), X.columns[indices[:n_features]], rotation=90)
            plt.tight_layout()
            
            # Salva o gráfico
            filename = f'fi-{dataset_name}-{corr_threshold}-{const_threshold}-{seed}.pdf'
            plt.savefig(f'{output_dir}/{filename}', format='pdf')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Erro ao plotar importância das features: {str(e)}")