# Sistema de Predição de Evasão Universitária

Este repositório contém scripts e análises para predição de taxas de evasão de estudantes universitários usando machine learning. Note que todos os arquivos Excel, CSV e Datasets originais são ignorados para upload neste repositório.

## Estrutura Original do Projeto

### `data-generator`
- **Descrição**: Contém scripts que geram várias versões do dataset original.
- **Objetivo**: Gerar várias versões do dataset original.

### `full-experiments`
- **Descrição**: Contém scripts que executam experimentos em todos os datasets.
- **Melhorias Necessárias**: Otimizar os scripts para evitar a execução de experimentos desnecessários, especialmente quando um dataset não possui equilíbrio adequado de classes.

### `periods-experiments`
- **Descrição**: Contém um script que executa predições específicas por períodos.
- **Objetivo**: Obter bons resultados para nossas análises.

## Sistema Unificado (Nova Implementação)

O projeto foi refatorado para unificar os módulos `full-experiments` e `periods-experiments` em um sistema único e mais eficiente.

### Estrutura do Sistema Unificado

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
└── example.py            # Exemplo de uso
```

### Funcionalidades

- **Processamento unificado**: Integra os módulos `full-experiments` e `periods-experiments` em um sistema único
- **Configuração centralizada**: Parâmetros e configurações em um único local
- **Validação robusta**: Verificação de dados de entrada e tratamento de erros
- **Processamento paralelo**: Execução eficiente de múltiplos experimentos
- **Resultados organizados**: Estrutura clara para armazenamento de resultados
- **Logging estruturado**: Registro detalhado de operações e erros

## Algoritmos Suportados

- Decision Tree (dt)
- Random Forest (rf)
- K-Neighbors (neigh)
- Naive Bayes (nb)

## Requisitos

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyyaml

## Uso Básico

```python
from ml_predictor import StudentDropoutPredictor

# Inicializa o preditor
predictor = StudentDropoutPredictor()

# Executa experimentos
results = predictor.run_experiments(
    datasets=['datasets/data_dv_last_occurence_FormReg_DesisTran.csv'],
    algorithms=['rf', 'dt'],
    experiment_type='full'  # ou 'periods'
)

# Salva os resultados
predictor.save_results(results, output_dir='results/')
```

## Linha de Comando

O sistema também pode ser executado via linha de comando usando o script `example.py`:

```bash
python example.py --datasets datasets/ --algorithms dt rf --experiment-type full
```

Opções disponíveis:

- `--datasets`: Caminhos para datasets ou diretórios (pode ser múltiplos)
- `--algorithms`: Algoritmos a serem utilizados (dt, rf, neigh, nb)
- `--experiment-type`: Tipo de experimento (full ou periods)
- `--n-jobs`: Número de processos paralelos
- `--output-dir`: Diretório para salvar resultados
- `--config`: Caminho para arquivo de configuração YAML

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

## Configuração Personalizada

É possível personalizar as configurações através de um arquivo YAML:

```yaml
# config.yaml
CORRELATION_THRESHOLDS: [0.8, 0.9]
CONSTANT_THRESHOLDS: [0.05, 0.1]
SEEDS: [42, 123]
IMPUTATION_STRATEGY: "median"
```

E carregar no sistema:

```python
predictor = StudentDropoutPredictor(config='config.yaml')
```

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

