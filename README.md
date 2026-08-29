# ARCHANGEL

ARCHANGEL é um pipeline local de pesquisa quantitativa para cripto perpétuos e, em uma fase posterior, futuros de commodities como coffee, cocoa e cotton.

O objetivo de pesquisa é construir um sistema realista, auditável e AI-friendly para estudar sinais, datasets, labels, treino walk-forward, backtest de portfólio, custos, risco, stress e registry de experimentos. A meta econômica de longo prazo é buscar retorno líquido anual acima de 20%, com controle explícito de drawdown, mas o código atual ainda é uma plataforma de pesquisa e validação. Ele não deve ser interpretado como aprovação para operar capital real.

## Estado Atual

O projeto roda localmente em `D:\1_ARCHANGEL` usando o Python próprio do projeto em:

```text
D:\1_ARCHANGEL\_PYTHON\Python314\python.exe
```

As etapas principais ficam em:

```text
D:\1_ARCHANGEL\0_REGRAS_MANDATO
```

Os JSONs AI-friendly de cada run são gerados localmente em:

```text
D:\1_ARCHANGEL\0_REGRAS_MANDATO\BASE_JSON
```

Esses JSONs são a mesa de controle do Codex para entender performance, ativos, datasets, labels, modelos, backtests, validações e registry. Por segurança e tamanho, os outputs gerados não são versionados por padrão neste repositório público.

## Pipeline

Orquestrador:

```text
0_REGRAS_MANDATO\00_RUN_ALL.py
```

Etapas atuais:

```text
0_DIAGNOSTICO_HARDWARE_BARRAMENTO.py
0_GERA_PYTHON_ENVIRONMENT_JSON.py
0_MAPEAMENTO_DIRETORIOS.py
1_MAPA_ATIVOS.py
2_BUSCA_DADOS.py
0_AUDITA_QUALIDADE_DADOS.py
3_GERA_FEATURES.py
4_GERA_LABELS.py
5_MONTA_DATASETS_ML.py
6_WALK_FORWARD_TRAINING.py
7_BACKTEST_PORTFOLIO.py
8_EXPERIMENT_REGISTRY.py
```

## Como Rodar Localmente

No PowerShell:

```powershell
cd D:\1_ARCHANGEL
.\_PYTHON\Python314\python.exe .\0_REGRAS_MANDATO\00_RUN_ALL.py
```

Também existe o atalho local:

```text
RUN_ALL_ARCHANGEL.cmd
```

No VS Code, a pasta `.vscode` mantém o terminal apontando para `D:\1_ARCHANGEL` e o interpretador Python do projeto.

## O Que Fica Fora do Git

Este repositório versiona o código e a estrutura de operação, não os dados pesados.

Ficam locais por padrão:

```text
_PYTHON
_CACHE
2_BASES
3_FEATURES
4_LABELS
5_DATASETS_ML
6_EXPERIMENTS
7_BACKTEST_PORTFOLIO
8_EXPERIMENT_REGISTRY
0_REGRAS_MANDATO\BASE_JSON
```

Isso evita publicar bases, features, labels, modelos, predições, backtests, JSONs com caminhos locais e documentos internos.

## Validação

O GitHub Actions faz uma validação leve do código:

- compila os `.py` versionados;
- verifica se arquivos pesados ou segredos óbvios não entraram no commit;
- não executa treino, download de dados, CUDA ou backtest pesado.

As validações profundas continuam sendo locais, usando os JSONs gerados no `BASE_JSON`.

## Próximos Passos

1. Manter o registry formal de experimentos como fonte mestre antes de qualquer comparação de hipóteses.
2. Expandir features não convencionais com ablação, walk-forward e stress de custos.
3. Preparar execução em testnet para Bybit, Binance e Kraken Pro, sem live trading nesta fase.
4. Evoluir o pipeline para futuros de commodities após a estabilidade em cripto perpétuos.

