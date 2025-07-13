#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para executar experimentos de predição de evasão escolar
usando o pacote ml_predictor.
"""

import os
import argparse
import logging
from ml_predictor import StudentDropoutPredictor, Config, setup_logging


def parse_arguments():
    """Analisa os argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='Executa experimentos de predição de evasão escolar'
    )
    
    parser.add_argument(
        '--datasets', '-d',
        type=str,
        required=True,
        help='Caminho para o diretório de datasets ou arquivo CSV específico'
    )
    
    parser.add_argument(
        '--algorithms', '-a',
        type=str,
        nargs='+',
        choices=list(Config.ALGORITHMS.keys()),
        default=list(Config.ALGORITHMS.keys()),
        help='Algoritmos a serem utilizados (default: todos)'
    )
    
    parser.add_argument(
        '--corr-thresholds', '-c',
        type=float,
        nargs='+',
        default=Config.CORRELATION_THRESHOLDS,
        help='Thresholds de correlação (default: Config.CORRELATION_THRESHOLDS)'
    )
    
    parser.add_argument(
        '--const-thresholds', '-t',
        type=float,
        nargs='+',
        default=Config.CONSTANT_THRESHOLDS,
        help='Thresholds de constantes (default: Config.CONSTANT_THRESHOLDS)'
    )
    
    parser.add_argument(
        '--seeds', '-s',
        type=int,
        nargs='+',
        default=Config.SEEDS,
        help='Seeds para reprodutibilidade (default: Config.SEEDS)'
    )
    
    parser.add_argument(
        '--experiment-type', '-e',
        type=str,
        choices=['full', 'periods'],
        default='full',
        help='Tipo de experimento (default: full)'
    )
    
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=None,
        help='Número de processos paralelos (default: CPU count - 1)'
    )
    
    parser.add_argument(
        '--config', '-f',
        type=str,
        default=None,
        help='Caminho para arquivo de configuração YAML'
    )
    
    return parser.parse_args()


def main():
    """Função principal"""
    # Configura o logger
    logger = setup_logging()
    logger.info("Iniciando execução de experimentos")
    
    # Analisa os argumentos
    args = parse_arguments()
    
    # Inicializa o preditor
    predictor = StudentDropoutPredictor(config=args.config)
    
    # Executa os experimentos
    results = predictor.run_experiments(
        datasets=args.datasets,
        algorithms=args.algorithms,
        corr_thresholds=args.corr_thresholds,
        const_thresholds=args.const_thresholds,
        seeds=args.seeds,
        experiment_type=args.experiment_type,
        n_jobs=args.n_jobs
    )
    
    # Exibe um resumo dos resultados
    logger.info(f"Total de {len(results)} experimentos concluídos com sucesso")
    
    if results:
        # Calcula a média das métricas principais
        avg_accuracy = sum(r['results']['accuracy'] for r in results) / len(results)
        avg_balanced_accuracy = sum(r['results']['balanced_accuracy'] for r in results) / len(results)
        avg_f1 = sum(r['results']['f1_score'] for r in results) / len(results)
        
        logger.info(f"Média de acurácia: {avg_accuracy:.4f}")
        logger.info(f"Média de acurácia balanceada: {avg_balanced_accuracy:.4f}")
        logger.info(f"Média de F1-score: {avg_f1:.4f}")
        
        # Exibe o melhor resultado
        best_result = max(results, key=lambda r: r['results']['f1_score'])
        logger.info("Melhor resultado (F1-score):")
        logger.info(f"  Dataset: {best_result['dataset_info']['name']}")
        logger.info(f"  Algoritmo: {best_result['experiment_config']['algorithm']}")
        logger.info(f"  Threshold de correlação: {best_result['experiment_config']['corr_threshold']}")
        logger.info(f"  Threshold de constantes: {best_result['experiment_config']['const_threshold']}")
        logger.info(f"  Seed: {best_result['experiment_config']['seed']}")
        logger.info(f"  Acurácia: {best_result['results']['accuracy']:.4f}")
        logger.info(f"  Acurácia balanceada: {best_result['results']['balanced_accuracy']:.4f}")
        logger.info(f"  F1-score: {best_result['results']['f1_score']:.4f}")
        logger.info(f"  Precisão: {best_result['results']['precision']:.4f}")
        logger.info(f"  Recall: {best_result['results']['recall']:.4f}")
        
        # Salva os resultados em arquivo
        csv_path, json_path, report_paths = predictor.save_results(results)
        
        # Informa onde os resultados foram salvos
        output_paths = []
        if csv_path:
            output_paths.append(csv_path)
        if json_path:
            output_paths.append(json_path)
        if report_paths:
            output_paths.extend(report_paths)
            
        if output_paths:
            logger.info("Arquivos gerados:")
            for path in output_paths:
                logger.info(f"  - {path}")
    
    logger.info("Execução concluída. Resultados salvos em: " + os.path.join(Config.RESULTS_DIR, 'experiments'))


if __name__ == '__main__':
    main()
