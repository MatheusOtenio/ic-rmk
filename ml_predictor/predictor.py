import os
import pandas as pd
import numpy as np
import json
import yaml
import multiprocessing
import time
from datetime import datetime
import logging
from pathlib import Path

from .config import Config
from .utils import setup_logging, create_directory_structure, validate_dataset, plot_class_distribution
from .processor import DataProcessor

class StudentDropoutPredictor:
    """Classe principal para predição de evasão de estudantes universitários"""
    
    def __init__(self, config=None):
        """Inicializa o preditor de evasão
        
        Args:
            config: Caminho para arquivo de configuração YAML ou dicionário de configuração
        """
        # Configura o logger
        self.logger = setup_logging()
        self.logger.info("Inicializando StudentDropoutPredictor")
        
        # Cria a estrutura de diretórios
        create_directory_structure()
        
        # Carrega configurações personalizadas se fornecidas
        if config:
            self._load_custom_config(config)
        
        # Inicializa o processador de dados
        self.processor = DataProcessor(logger=self.logger)
    
    def _load_custom_config(self, config):
        """Carrega configurações personalizadas"""
        try:
            if isinstance(config, str):
                # Carrega de arquivo YAML
                if os.path.exists(config):
                    with open(config, 'r') as f:
                        custom_config = yaml.safe_load(f)
                    
                    self.logger.info(f"Carregando configurações de {config}")
                    
                    # Atualiza as configurações
                    for key, value in custom_config.items():
                        if hasattr(Config, key):
                            setattr(Config, key, value)
                            self.logger.debug(f"Configuração atualizada: {key} = {value}")
                else:
                    self.logger.warning(f"Arquivo de configuração não encontrado: {config}")
            
            elif isinstance(config, dict):
                # Carrega do dicionário
                self.logger.info("Carregando configurações do dicionário")
                
                # Atualiza as configurações
                for key, value in config.items():
                    if hasattr(Config, key):
                        setattr(Config, key, value)
                        self.logger.debug(f"Configuração atualizada: {key} = {value}")
            
            else:
                self.logger.warning(f"Formato de configuração não suportado: {type(config)}")
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar configurações: {str(e)}")
    
    def run_experiments(self, datasets, algorithms=None, corr_thresholds=None, const_thresholds=None, 
                        seeds=None, experiment_type='full', n_jobs=None):
        """Executa experimentos de predição de evasão
        
        Args:
            datasets: Lista de caminhos para datasets ou diretório contendo datasets
            algorithms: Lista de algoritmos a serem utilizados (default: todos)
            corr_thresholds: Lista de thresholds de correlação (default: Config.CORRELATION_THRESHOLDS)
            const_thresholds: Lista de thresholds de constantes (default: Config.CONSTANT_THRESHOLDS)
            seeds: Lista de seeds para reprodutibilidade (default: Config.SEEDS)
            experiment_type: Tipo de experimento ('full' ou 'periods')
            n_jobs: Número de processos paralelos (default: CPU count - 1)
            
        Returns:
            Lista de resultados dos experimentos
        """
        start_time = time.time()
        self.logger.info(f"Iniciando experimentos do tipo '{experiment_type}'")
        
        # Configura parâmetros padrão
        algorithms = algorithms or list(Config.ALGORITHMS.keys())
        corr_thresholds = corr_thresholds or Config.CORRELATION_THRESHOLDS
        const_thresholds = const_thresholds or Config.CONSTANT_THRESHOLDS
        seeds = seeds or Config.SEEDS
        n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        
        # Verifica e carrega os datasets
        dataset_paths = self._resolve_dataset_paths(datasets)
        if not dataset_paths:
            self.logger.error("Nenhum dataset encontrado")
            return []
        
        # Gera a grade de experimentos
        experiments = self._generate_experiment_grid(dataset_paths, algorithms, corr_thresholds, const_thresholds, seeds, experiment_type)
        self.logger.info(f"Total de {len(experiments)} experimentos a serem executados")
        
        # Executa os experimentos em paralelo
        results = []
        errors = []
        
        if n_jobs > 1:
            self.logger.info(f"Executando experimentos em paralelo com {n_jobs} processos")
            with multiprocessing.Pool(processes=n_jobs) as pool:
                experiment_results = pool.map(self._execute_experiment_wrapper, experiments)
                
                # Processa os resultados
                for result in experiment_results:
                    if 'error' in result:
                        errors.append(result)
                    else:
                        results.append(result)
        else:
            self.logger.info("Executando experimentos sequencialmente")
            for experiment in experiments:
                result = self._execute_experiment_wrapper(experiment)
                if 'error' in result:
                    errors.append(result)
                else:
                    results.append(result)
        
        # Salva os erros
        if errors:
            self._save_errors(errors)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experimentos concluídos em {elapsed_time:.2f} segundos. {len(results)} bem-sucedidos, {len(errors)} com erro.")
        
        return results
    
    def _resolve_dataset_paths(self, datasets):
        """Resolve os caminhos dos datasets"""
        dataset_paths = []
        
        # Caso seja uma string única, converte para lista
        if isinstance(datasets, str):
            datasets = [datasets]
        
        for dataset in datasets:
            # Verifica se é um caminho de arquivo
            if os.path.isfile(dataset) and dataset.endswith('.csv'):
                dataset_paths.append(dataset)
            
            # Verifica se é um diretório
            elif os.path.isdir(dataset):
                # Adiciona todos os CSVs do diretório
                for file in os.listdir(dataset):
                    if file.endswith('.csv'):
                        dataset_paths.append(os.path.join(dataset, file))
        
        return dataset_paths
    
    def _generate_experiment_grid(self, dataset_paths, algorithms, corr_thresholds, const_thresholds, seeds, experiment_type):
        """Gera a grade de experimentos"""
        experiments = []
        
        for dataset_path in dataset_paths:
            # Carrega o dataset para validação
            try:
                data, dataset_name = self.processor.load_dataset(dataset_path)
                
                # Verifica se o dataset é válido para o tipo de experimento
                if experiment_type == 'periods' and 'periodo' not in dataset_name.lower():
                    self.logger.info(f"Pulando dataset {dataset_name} para experimento do tipo 'periods' (não contém 'periodo' no nome)")
                    continue
                
                # Valida o dataset
                validate_dataset(data, dataset_name, self.logger)
                
                # Plota a distribuição de classes
                plot_class_distribution(data, dataset_name)
                
                # Gera os experimentos para este dataset
                for algorithm in algorithms:
                    for corr_threshold in corr_thresholds:
                        for const_threshold in const_thresholds:
                            for seed in seeds:
                                experiment = {
                                    'dataset_path': dataset_path,
                                    'dataset_name': dataset_name,
                                    'algorithm': algorithm,
                                    'corr_threshold': corr_threshold,
                                    'const_threshold': const_threshold,
                                    'seed': seed,
                                    'experiment_type': experiment_type
                                }
                                experiments.append(experiment)
            
            except Exception as e:
                self.logger.error(f"Erro ao processar dataset {dataset_path}: {str(e)}")
                # Registra o erro, mas continua com os próximos datasets
                continue
        
        return experiments
    
    def _execute_experiment_wrapper(self, experiment):
        """Wrapper para execução de experimentos (compatível com multiprocessing)"""
        try:
            # Extrai os parâmetros do experimento
            dataset_path = experiment['dataset_path']
            dataset_name = experiment['dataset_name']
            algorithm = experiment['algorithm']
            corr_threshold = experiment['corr_threshold']
            const_threshold = experiment['const_threshold']
            seed = experiment['seed']
            
            # Carrega o dataset
            data, _ = self.processor.load_dataset(dataset_path)
            
            # Executa o experimento
            result = self.processor.execute(data, dataset_name, algorithm, corr_threshold, const_threshold, seed)
            
            # Adiciona informações do experimento ao resultado
            result['experiment_id'] = f"{dataset_name}_{algorithm}_{corr_threshold}_{const_threshold}_{seed}"
            
            return result
        
        except Exception as e:
            # Captura qualquer erro e retorna como parte do resultado
            error_result = {
                'error': str(e),
                'dataset_path': experiment.get('dataset_path', 'unknown'),
                'dataset_name': experiment.get('dataset_name', 'unknown'),
                'algorithm': experiment.get('algorithm', 'unknown'),
                'corr_threshold': experiment.get('corr_threshold', 'unknown'),
                'const_threshold': experiment.get('const_threshold', 'unknown'),
                'seed': experiment.get('seed', 'unknown'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return error_result
    
    def save_results(self, results, output_dir=None):
        """Salva os resultados dos experimentos"""
        if not results:
            self.logger.warning("Nenhum resultado para salvar")
            return
        
        # Define o diretório de saída
        if output_dir is None:
            output_dir = Config.EXPERIMENTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Gera um timestamp para o nome do arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva os resultados em formato CSV
        csv_results = []
        for result in results:
            # Extrai informações básicas
            dataset_info = result.get('dataset_info', {})
            experiment_config = result.get('experiment_config', {})
            metrics = result.get('results', {})
            
            # Cria uma linha para o CSV
            csv_row = {
                'dataset': dataset_info.get('name', 'unknown'),
                'samples': dataset_info.get('samples', 0),
                'features': dataset_info.get('features', 0),
                'algorithm': experiment_config.get('algorithm', 'unknown'),
                'corr_threshold': experiment_config.get('corr_threshold', 0),
                'const_threshold': experiment_config.get('const_threshold', 0),
                'seed': experiment_config.get('seed', 0),
                'accuracy': metrics.get('accuracy', 0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'f1_score': metrics.get('f1_score', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'experiment_id': result.get('experiment_id', 'unknown')
            }
            
            csv_results.append(csv_row)
        
        # Salva o CSV
        csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
        pd.DataFrame(csv_results).to_csv(csv_path, index=False)
        self.logger.info(f"Resultados salvos em {csv_path}")
        
        # Salva as predições
        predictions_dir = Config.PREDICTIONS_DIR
        os.makedirs(predictions_dir, exist_ok=True)
        
        for result in results:
            # Extrai informações
            dataset_name = result.get('dataset_info', {}).get('name', 'unknown')
            algorithm = result.get('experiment_config', {}).get('algorithm', 'unknown')
            corr_threshold = result.get('experiment_config', {}).get('corr_threshold', 0)
            const_threshold = result.get('experiment_config', {}).get('const_threshold', 0)
            seed = result.get('experiment_config', {}).get('seed', 0)
            
            # Extrai predições
            predictions = result.get('predictions', {})
            true_labels = predictions.get('true_labels', [])
            predicted_labels = predictions.get('predicted_labels', [])
            probabilities = predictions.get('probabilities', [])
            
            # Cria DataFrame de predições
            pred_df = pd.DataFrame({
                'true_label': true_labels,
                'predicted_label': predicted_labels
            })
            
            # Adiciona probabilidades se disponíveis
            if probabilities:
                pred_df['probability'] = probabilities
            
            # Salva as predições
            pred_filename = f"pred_{dataset_name}_{algorithm}_{corr_threshold}_{const_threshold}_{seed}.csv"
            pred_path = os.path.join(predictions_dir, pred_filename)
            pred_df.to_csv(pred_path, index=False)
        
        # Salva os resultados completos em JSON
        json_path = os.path.join(output_dir, f"full_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Resultados completos salvos em {json_path}")
        
        return csv_path, json_path
    
    def _save_errors(self, errors):
        """Salva os erros encontrados durante os experimentos"""
        if not errors:
            return
        
        # Define o diretório de logs
        logs_dir = Config.LOGS_DIR
        os.makedirs(logs_dir, exist_ok=True)
        
        # Gera um timestamp para o nome do arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva os erros em formato CSV
        errors_df = pd.DataFrame(errors)
        errors_path = os.path.join(logs_dir, f"errors_{timestamp}.csv")
        errors_df.to_csv(errors_path, index=False)
        
        self.logger.info(f"{len(errors)} erros salvos em {errors_path}")