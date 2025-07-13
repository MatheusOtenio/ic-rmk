#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemplo de uso do sistema unificado de predição de evasão universitária.

Este script demonstra como utilizar o pacote ml_predictor para executar
experimentos de predição de evasão universitária.
"""

import os
import sys
import argparse
from ml_predictor import StudentDropoutPredictor

def parse_arguments():
    """Processa os argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Sistema de Predição de Evasão Universitária')
    
    parser.add_argument('--datasets', nargs='+', default=['datasets'],
                        help='Caminhos para datasets ou diretórios contendo datasets')
    
    parser.add_argument('--algorithms', nargs='+', default=['dt', 'rf', 'neigh', 'nb'],
                        help='Algoritmos a serem utilizados (dt, rf, neigh, nb)')
    
    parser.add_argument('--experiment-type', choices=['full', 'periods'], default='full',
                        help='Tipo de experimento (full ou periods)')
    
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Número de processos paralelos')
    
    parser.add_argument('--output-dir', default='results',
                        help='Diretório para salvar os resultados')
    
    parser.add_argument('--config', default=None,
                        help='Caminho para arquivo de configuração YAML')
    
    return parser.parse_args()

def main():
    """Função principal"""
    # Processa os argumentos
    args = parse_arguments()
    
    # Inicializa o preditor
    predictor = StudentDropoutPredictor(config=args.config)
    
    # Executa os experimentos
    results = predictor.run_experiments(
        datasets=args.datasets,
        algorithms=args.algorithms,
        experiment_type=args.experiment_type,
        n_jobs=args.n_jobs
    )
    
    # Salva os resultados
    if results:
        csv_path, json_path = predictor.save_results(results, output_dir=args.output_dir)
        print(f"\nResultados salvos em:\n- {csv_path}\n- {json_path}")
    else:
        print("\nNenhum resultado gerado. Verifique os logs para mais informações.")

if __name__ == '__main__':
    main()