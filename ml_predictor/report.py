import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from .config import Config
import json
import base64
from io import BytesIO

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

class ReportGenerator:
    """Classe para geração de relatórios consolidados"""
    
    def __init__(self, logger=None):
        """Inicializa o gerador de relatórios"""
        self.logger = logger
        self.results_dir = Config.RESULTS_DIR
        self.experiments_dir = Config.EXPERIMENTS_DIR
        self.plots_dir = Config.PLOTS_DIR
        
        # Cria diretório para relatórios consolidados
        self.reports_dir = 'results/reports'
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_consolidated_report(self, results_df):
        """Gera um relatório consolidado com os resultados dos experimentos"""
        if not Config.GENERATE_CONSOLIDATED_REPORT:
            return []
            
        try:
            # Verifica se há resultados para gerar o relatório
            if results_df is None or len(results_df) == 0:
                if self.logger:
                    self.logger.warning("Não há resultados para gerar o relatório consolidado")
                return []
                
            # Cria timestamp para o relatório
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.reports_dir}/report_{timestamp}"
            
            # Gera o relatório em todos os formatos habilitados
            report_filenames = self._create_report_figure(results_df, base_filename)
            
            if self.logger and report_filenames:
                self.logger.info(f"Relatório(s) consolidado(s) gerado(s): {', '.join(report_filenames)}")
                
            return report_filenames
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erro ao gerar relatório consolidado: {str(e)}")
            return []
    
    def _create_report_figure(self, results_df, base_filename):
        """Cria a figura do relatório consolidado"""
        # Configura o tamanho da figura baseado na quantidade de algoritmos e datasets
        n_algorithms = results_df['algorithm'].nunique()
        n_datasets = results_df['dataset'].nunique()
        
        # Cria uma figura grande o suficiente para acomodar todos os gráficos
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Relatório Consolidado de Experimentos', fontsize=16)
        
        # 1. Gráfico de barras com métricas por algoritmo
        plt.subplot(2, 2, 1)
        metrics_by_algo = results_df.groupby('algorithm')[['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']].mean()
        metrics_by_algo.plot(kind='bar', ax=plt.gca())
        plt.title('Métricas Médias por Algoritmo')
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 2. Boxplot de F1-score por algoritmo
        plt.subplot(2, 2, 2)
        sns.boxplot(x='algorithm', y='f1', data=results_df)
        plt.title('Distribuição de F1-Score por Algoritmo')
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 3. Heatmap de correlação entre métricas
        plt.subplot(2, 2, 3)
        corr = results_df[['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlação entre Métricas')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 4. Tabela com os melhores resultados
        plt.subplot(2, 2, 4)
        best_results = results_df.sort_values('f1', ascending=False).head(5)
        cell_text = []
        for _, row in best_results.iterrows():
            cell_text.append([row['dataset'], row['algorithm'], 
                             f"{row['f1']:.4f}", f"{row['accuracy']:.4f}", 
                             f"{row['precision']:.4f}", f"{row['recall']:.4f}"])
        
        plt.axis('off')
        table = plt.table(cellText=cell_text,
                          colLabels=['Dataset', 'Algoritmo', 'F1', 'Accuracy', 'Precision', 'Recall'],
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.title('Top 5 Melhores Resultados', y=1.1)
        
        # Ajusta o layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Lista para armazenar os caminhos dos arquivos salvos
        saved_files = []
        
        # Salva a figura em todos os formatos habilitados
        for fmt, enabled in Config.OUTPUT_FORMATS.items():
            if enabled and fmt in ['pdf', 'png', 'svg']:
                filename = f"{base_filename}.{fmt}"
                if fmt in ['png', 'jpg']:
                    plt.savefig(filename, format=fmt, dpi=Config.PLOT_DPI)
                else:
                    plt.savefig(filename, format=fmt)
                saved_files.append(filename)
                
        plt.close()
        return saved_files
    
    def generate_experiment_summary(self, results_df, output_path=None):
        """Gera um resumo dos experimentos em formato JSON"""
        if not Config.GENERATE_CONSOLIDATED_REPORT:
            return []
            
        try:
            if results_df is None or len(results_df) == 0:
                if self.logger:
                    self.logger.warning("Não há resultados para gerar o resumo dos experimentos")
                return []
                
            # Define o caminho de saída
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{self.reports_dir}/summary_{timestamp}.json"
            
            # Calcula estatísticas gerais
            summary = {
                "total_experiments": len(results_df),
                "datasets": results_df['dataset'].nunique(),
                "algorithms": results_df['algorithm'].unique().tolist(),
                "average_metrics": {
                    "accuracy": results_df['accuracy'].mean(),
                    "balanced_accuracy": results_df['balanced_accuracy'].mean(),
                    "f1": results_df['f1'].mean(),
                    "precision": results_df['precision'].mean(),
                    "recall": results_df['recall'].mean()
                },
                "best_experiment": {}
            }
            
            # Encontra o melhor experimento baseado no F1-score
            best_idx = results_df['f1'].idxmax()
            best_exp = results_df.loc[best_idx]
            
            summary["best_experiment"] = {
                "dataset": best_exp['dataset'],
                "algorithm": best_exp['algorithm'],
                "metrics": {
                    "accuracy": best_exp['accuracy'],
                    "balanced_accuracy": best_exp['balanced_accuracy'],
                    "f1": best_exp['f1'],
                    "precision": best_exp['precision'],
                    "recall": best_exp['recall']
                },
                "parameters": {
                    "corr_threshold": best_exp['corr_threshold'],
                    "const_threshold": best_exp['const_threshold'],
                    "seed": best_exp['seed']
                }
            }
            
            # Adiciona métricas por algoritmo
            summary["metrics_by_algorithm"] = {}
            for algo in results_df['algorithm'].unique():
                algo_df = results_df[results_df['algorithm'] == algo]
                summary["metrics_by_algorithm"][algo] = {
                    "accuracy": algo_df['accuracy'].mean(),
                    "balanced_accuracy": algo_df['balanced_accuracy'].mean(),
                    "f1": algo_df['f1'].mean(),
                    "precision": algo_df['precision'].mean(),
                    "recall": algo_df['recall'].mean()
                }
            
            output_files = []
            
            # Salva o resumo em JSON se configurado
            if Config.OUTPUT_FORMATS.get('json', False):
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=4, default=convert_numpy_types)
                    
                if self.logger:
                    self.logger.info(f"Resumo dos experimentos gerado em JSON: {output_path}")
                output_files.append(output_path)
                
            # Gera relatório HTML se configurado
            if Config.GENERATE_HTML_REPORT and Config.OUTPUT_FORMATS.get('html', False):
                html_path = self.generate_html_report(results_df, summary)
                if html_path:
                    if self.logger:
                        self.logger.info(f"Relatório HTML gerado: {html_path}")
                    output_files.append(html_path)
                
            return output_files
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erro ao gerar resumo dos experimentos: {str(e)}")
            return []
                
    def generate_html_report(self, results_df, summary=None):
        """Gera um relatório HTML com os resultados dos experimentos"""
        if not Config.GENERATE_HTML_REPORT or not Config.OUTPUT_FORMATS.get('html', False):
            return
            
        try:
            if results_df is None or len(results_df) == 0:
                if self.logger:
                    self.logger.warning("Não há resultados para gerar o relatório HTML")
                return
                
            # Define o caminho de saída
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.reports_dir}/report_{timestamp}.html"
            
            # Gera gráficos para o relatório HTML
            fig_metrics = plt.figure(figsize=(10, 6))
            metrics_by_algo = results_df.groupby('algorithm')[['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']].mean()
            metrics_by_algo.plot(kind='bar')
            plt.title('Métricas Médias por Algoritmo')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Converte o gráfico para base64 para incluir no HTML
            buf = BytesIO()
            fig_metrics.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig_metrics)
            
            # Gera boxplot de F1-score
            fig_boxplot = plt.figure(figsize=(10, 6))
            sns.boxplot(x='algorithm', y='f1', data=results_df)
            plt.title('Distribuição de F1-Score por Algoritmo')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf2 = BytesIO()
            fig_boxplot.savefig(buf2, format='png')
            buf2.seek(0)
            img_str2 = base64.b64encode(buf2.read()).decode('utf-8')
            plt.close(fig_boxplot)
            
            # Cria o conteúdo HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Relatório de Experimentos - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-value {{ font-weight: bold; }}
                    .container {{ margin-bottom: 30px; }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>Relatório de Experimentos de Predição de Evasão</h1>
                <div class="container">
                    <h2>Resumo dos Experimentos</h2>
                    <table>
                        <tr>
                            <th>Total de Experimentos</th>
                            <td class="metric-value">{summary['total_experiments']}</td>
                        </tr>
                        <tr>
                            <th>Datasets</th>
                            <td>{summary['datasets']}</td>
                        </tr>
                        <tr>
                            <th>Algoritmos</th>
                            <td>{', '.join(summary['algorithms'])}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="container">
                    <h2>Métricas Médias</h2>
                    <table>
                        <tr>
                            <th>Acurácia</th>
                            <td class="metric-value">{summary['average_metrics']['accuracy']:.4f}</td>
                        </tr>
                        <tr>
                            <th>Acurácia Balanceada</th>
                            <td class="metric-value">{summary['average_metrics']['balanced_accuracy']:.4f}</td>
                        </tr>
                        <tr>
                            <th>F1-Score</th>
                            <td class="metric-value">{summary['average_metrics']['f1']:.4f}</td>
                        </tr>
                        <tr>
                            <th>Precisão</th>
                            <td class="metric-value">{summary['average_metrics']['precision']:.4f}</td>
                        </tr>
                        <tr>
                            <th>Recall</th>
                            <td class="metric-value">{summary['average_metrics']['recall']:.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="container">
                    <h2>Melhor Experimento</h2>
                    <table>
                        <tr>
                            <th>Dataset</th>
                            <td>{summary['best_experiment']['dataset']}</td>
                        </tr>
                        <tr>
                            <th>Algoritmo</th>
                            <td>{summary['best_experiment']['algorithm']}</td>
                        </tr>
                        <tr>
                            <th>F1-Score</th>
                            <td class="metric-value">{summary['best_experiment']['metrics']['f1']:.4f}</td>
                        </tr>
                        <tr>
                            <th>Acurácia</th>
                            <td class="metric-value">{summary['best_experiment']['metrics']['accuracy']:.4f}</td>
                        </tr>
                        <tr>
                            <th>Precisão</th>
                            <td class="metric-value">{summary['best_experiment']['metrics']['precision']:.4f}</td>
                        </tr>
                        <tr>
                            <th>Recall</th>
                            <td class="metric-value">{summary['best_experiment']['metrics']['recall']:.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="container">
                    <h2>Visualizações</h2>
                    <div class="chart">
                        <h3>Métricas por Algoritmo</h3>
                        <img src="data:image/png;base64,{img_str}" alt="Métricas por Algoritmo">
                    </div>
                    <div class="chart">
                        <h3>Distribuição de F1-Score</h3>
                        <img src="data:image/png;base64,{img_str2}" alt="Distribuição de F1-Score">
                    </div>
                </div>
                
                <div class="container">
                    <h2>Resultados Detalhados</h2>
                    <table>
                        <tr>
                            <th>Dataset</th>
                            <th>Algoritmo</th>
                            <th>F1-Score</th>
                            <th>Acurácia</th>
                            <th>Precisão</th>
                            <th>Recall</th>
                        </tr>
            """
            
            # Adiciona linhas da tabela para cada experimento
            for _, row in results_df.iterrows():
                html_content += f"""
                        <tr>
                            <td>{row['dataset']}</td>
                            <td>{row['algorithm']}</td>
                            <td class="metric-value">{row['f1']:.4f}</td>
                            <td>{row['accuracy']:.4f}</td>
                            <td>{row['precision']:.4f}</td>
                            <td>{row['recall']:.4f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
                
                <div class="container">
                    <p><em>Relatório gerado em: {}</em></p>
                </div>
            </body>
            </html>
            """.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            
            # Salva o relatório HTML
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return output_path
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erro ao gerar relatório HTML: {str(e)}")
            return None