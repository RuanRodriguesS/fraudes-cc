# Detecção de Fraudes em Transações com Cartão de Crédito

Análise e detecção de fraudes em transações de cartão de crédito usando Machine Learning. O dataset utilizado é o [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) do Kaggle.

---

## Dataset

O dataset contém **1.296.675 transações** com as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `trans_date_trans_time` | Data e hora da transação |
| `cc_num` | Número do cartão de crédito |
| `merchant` | Nome do comerciante |
| `category` | Categoria da transação |
| `amt` | Valor da transação (USD) |
| `first` / `last` | Nome do titular do cartão |
| `gender` | Gênero do titular |
| `street` / `city` / `state` / `zip` | Endereço do titular |
| `lat` / `long` | Coordenadas geográficas do titular |
| `city_pop` | População da cidade |
| `job` | Profissão do titular |
| `dob` | Data de nascimento |
| `trans_num` | Número único da transação |
| `unix_time` | Timestamp UNIX |
| `merch_lat` / `merch_long` | Coordenadas do comerciante |
| `is_fraud` | Target — 0 = legítima, 1 = fraude |

### Distribuição das classes

| Classe | Transações | Percentual |
|---|---|---|
| Legítimas (0) | 1.289.169 | 99,42% |
| Fraudulentas (1) | 7.506 | 0,58% |

O dataset é bastante desbalanceado: menos de 1% das transações são fraudes.

---

## Análise Exploratória

O dataset possui 14 categorias de transação. Entre elas, as categorias online (`shopping_net`, `misc_net`) apresentam taxas de fraude proporcionalmente maiores que as presenciais (POS).

A análise temporal mostrou que fraudes tendem a ocorrer em horários específicos do dia e em determinados dias da semana.

Como engenharia de feature, foi calculada a distância em km entre o endereço do titular e o comerciante usando a **fórmula de Haversine** (`distance_km`). Transações fraudulentas costumam apresentar distâncias maiores.

---

## Modelo

O classificador usado foi o **XGBoost**, otimizado com **GridSearchCV**. O pipeline inclui:

- One-Hot Encoding na coluna `category`
- Balanceamento com **SMOTE** para compensar o desbalanceamento das classes

As principais métricas de avaliação foram **F1-Score**, **Precision**, **Recall** e **ROC-AUC**, mais adequadas para datasets desbalanceados do que a acurácia simples.

---

## Tecnologias

- `pandas` / `numpy` — manipulação de dados
- `matplotlib` / `seaborn` — visualizações
- `scikit-learn` — pipelines e métricas
- `imbalanced-learn` — SMOTE
- `xgboost` — classificador
- `kagglehub` — download do dataset

---

## Como executar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost kagglehub
```

Configure suas credenciais do Kaggle em `~/.kaggle/kaggle.json`, depois abra o notebook:

```bash
jupyter notebook fraudes.ipynb
```

O dataset é baixado automaticamente na primeira execução.

---

## Autor

**Ruan Rodrigues S.** — [GitHub](https://github.com/RuanRodriguesS)
