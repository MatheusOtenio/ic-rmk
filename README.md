# Sistema de Predição de Evasão Universitária

Este repositório contém scripts e análises para predição de taxas de evasão de estudantes universitários usando machine learning. Note que todos os arquivos Excel, CSV e Datasets originais são ignorados para upload neste repositório.

## Estrutura do Projeto

### `data-generator`
- **Descrição**: Contém scripts que geram várias versões do dataset original.
- **Objetivo**: Gerar várias versões do dataset original a partir de dados brutos.

### `ml_predictor`
- **Descrição**: Pacote principal que unifica a execução de experimentos de predição.
- **Objetivo**: Fornecer uma interface unificada e eficiente para experimentos de machine learning.

### Estrutura do Sistema

```
├── data-generator/       # Geração de datasets
├── datasets/             # Datasets processados
├── ml_predictor/         # Pacote principal
│   ├── __init__.py       # Inicialização do pacote
│   ├── config.py         # Configurações centralizadas
│   ├── predictor.py      # Classe principal
│   ├── processor.py      # Processamento de dados e experimentos
│   └── utils.py          # Funções utilitárias
├── results/              # Resultados dos experimentos
│   ├── experiments/      # Métricas e resultados
│   ├── predictions/      # Predições detalhadas
│   ├── plots/            # Visualizações
│   └── logs/             # Logs de execução
├── run_experiments.py    # Script de exemplo para execução de experimentos
└── requirements.txt      # Dependências do projeto
```

### Funcionalidades

- **Processamento unificado**: Sistema integrado para experimentos de machine learning
- **Configuração centralizada**: Parâmetros e configurações em um único local
- **Validação robusta**: Verificação de dados de entrada e tratamento de erros
- **Processamento paralelo**: Execução eficiente de múltiplos experimentos
- **Resultados organizados**: Estrutura clara para armazenamento de resultados
- **Logging estruturado**: Registro detalhado de operações e erros

## Como Usar

```bash
python run_experiments.py --datasets datasets/ --algorithms dt rf --corr-thresholds 0.8 0.9 --const-thresholds 0.1 0.2 --seeds 145 278
```

### Instalação

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

### Preparação de Dados

1. Coloque seus arquivos CSV no diretório `datasets/`
2. Ou use o script `data-generator/data-creator.py` para gerar datasets a partir de dados brutos

### Execução de Experimentos

Você pode executar experimentos usando o script `run_experiments.py`:

```bash
python run_experiments.py --datasets datasets/ --experiment-type full
```

Opções disponíveis:

- `--datasets`, `-d`: Caminho para o diretório de datasets ou arquivo CSV específico (obrigatório)
- `--algorithms`, `-a`: Algoritmos a serem utilizados (default: todos)
- `--corr-thresholds`, `-c`: Thresholds de correlação (default: Config.CORRELATION_THRESHOLDS)
- `--const-thresholds`, `-t`: Thresholds de constantes (default: Config.CONSTANT_THRESHOLDS)
- `--seeds`, `-s`: Seeds para reprodutibilidade (default: Config.SEEDS)
- `--experiment-type`, `-e`: Tipo de experimento ('full' ou 'periods', default: 'full')
- `--n-jobs`, `-j`: Número de processos paralelos (default: CPU count - 1)
- `--config`, `-f`: Caminho para arquivo de configuração YAML

### Uso Programático

Você também pode usar o pacote `ml_predictor` diretamente em seus scripts Python:

```python
from ml_predictor import StudentDropoutPredictor

# Inicializa o preditor
predictor = StudentDropoutPredictor()

# Executa experimentos
results = predictor.run_experiments(
    datasets='datasets/',
    algorithms=['dt', 'rf'],
    experiment_type='full'
)

# Processa os resultados
for result in results:
    print(f"Dataset: {result['dataset_name']}, Algoritmo: {result['algorithm']}, F1: {result['f1']:.4f}")
```

### Resultados

Os resultados dos experimentos são salvos no diretório `results/`:

- `results/experiments/`: Métricas e resultados em formato CSV
- `results/predictions/`: Predições detalhadas para cada experimento
- `results/plots/`: Visualizações (distribuição de classes, matrizes de confusão, etc.)
- `results/logs/`: Logs de execução

## Algoritmos Suportados

O sistema suporta os seguintes algoritmos de machine learning:

- **Decision Tree (dt)**: Árvore de decisão, útil para visualizar o processo de tomada de decisão
- **Random Forest (rf)**: Conjunto de árvores de decisão, geralmente com melhor desempenho
- **K-Neighbors (neigh)**: Classificação baseada em vizinhos mais próximos
- **Naive Bayes (nb)**: Classificador probabilístico baseado no teorema de Bayes

## Configuração

As configurações padrão estão definidas no arquivo `ml_predictor/config.py`. Você pode personalizar as configurações de duas maneiras:

1. **Arquivo YAML**: Crie um arquivo YAML com as configurações desejadas e passe-o para o construtor do `StudentDropoutPredictor`
2. **Dicionário**: Passe um dicionário com as configurações desejadas para o construtor do `StudentDropoutPredictor`

Exemplo de arquivo YAML:

```yaml
# config.yaml
CORRELATION_THRESHOLDS: [0.8, 0.9]
CONSTANT_THRESHOLDS: [0.1, 0.2]
SEEDS: [42, 123, 456]
IMPUTATION_STRATEGY: 'median'
```

Uso:

```python
predictor = StudentDropoutPredictor(config='config.yaml')
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença

Este projeto está licenciado sob a licença MIT.



## Formato dos Resultados

O sistema gera resultados no seguinte formato:

```json
{
    "dataset_info": {
        "name": "nome_do_dataset",
        "samples": 1000,
        "features": 20,
        "class_distribution": {"0": 800, "1": 200}
    },
    "experiment_config": {
        "algorithm": "Random Forest",
        "corr_threshold": 0.8,
        "const_threshold": 0.05,
        "seed": 42
    },
    "results": {
        "accuracy": 0.85,
        "balanced_accuracy": 0.82,
        "f1_score": 0.75,
        "precision": 0.80,
        "recall": 0.70,
        "confusion_matrix": [[700, 100], [50, 150]]
    },
    "predictions": {
        "true_labels": [0, 1, 0, ...],
        "predicted_labels": [0, 1, 1, ...],
        "probabilities": [0.2, 0.8, 0.6, ...]
    }
}
```

## Tratamento de Erros

O sistema implementa tratamento robusto para os seguintes erros:

- `FileNotFoundError`: Dados ausentes
- `ValueError`: Dados mal formatados ou inválidos
- `KeyError`: Colunas ausentes
- `MemoryError`: Recursos insuficientes
- `RuntimeError`: Falhas no processamento



## Troubleshooting

### Problemas Comuns

1. **Datasets não encontrados**
   - Verifique se os caminhos estão corretos
   - Execute o script `data-generator/data-creator.py` para gerar os datasets

2. **Erro de coluna 'target' ausente**
   - Verifique se os datasets possuem a coluna 'target'
   - Utilize datasets gerados pelo script `data-creator.py`

3. **Desbalanceamento severo de classes**
   - O sistema emite avisos para datasets com desbalanceamento
   - Considere técnicas de balanceamento ou ajuste de thresholds

4. **Erros de memória**
   - Reduza o número de processos paralelos (`--n-jobs`)
   - Processe datasets menores ou reduza o número de experimentos

## Contribuição

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Faça commit das mudanças (`git commit -am 'Adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

