# Sentimentanalys av VD-ord – Uppsatskod

Detta repo innehåller Python-kod som använts inom ramen för ett uppsatsarbete. Syftet är att analysera sentiment i svenska börsbolags VD-ord och undersöka samband med finansiell hälsa (Altman’s Z-score).

## Översikt

Koden består av två huvudsakliga analysvägar:

1. **FinBERT (via Hugging Face Transformers)** – Finjusterad BERT-modell för finansiell sentimentanalys.
2. **OpenAI GPT (via OpenAI API)** – Sentimentklassificering med hjälp av GPT-4.

---

## Förutsättningar

Innan du kör koden måste du säkerställa följande:

### 1. Dataset

Du behöver en CSV-fil med följande struktur:

```csv
company;year;letter
Boliden;2023;What is your experience of 2023?...
Boliden;2022;"Boliden is reporting historically...
Boliden;2021;The metals we produce are absol...
Boliden;2020;A year of very different chal...
Boliden;2019;Boliden has a broad port...
```

### 2. Installation av beroenden

#### FinBERT – Transformers
```pip install transformers torch pandas```

#### OpenAI – GPT-modell
```pip install openai pandas```


### 4. Användning
Ladda in din CSV, kör sentimentanalysen och exportera resultatet för vidare statistisk analys.
