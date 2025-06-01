from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

print(model.config._name_or_path)

def split_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def analyze_text(text):
    chunks = split_text(text)
    results = []

    for chunk in chunks:
        inputs = tokenizer.decode(chunk, skip_special_tokens=True)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            result = {
                "positive": probabilities[0][0].item(),
                "neutral": probabilities[0][1].item(),
                "negative": probabilities[0][2].item()
            }
            results.append(result)

    return results

def summarize_results(results):
    avg_positive = sum([res['positive'] for res in results]) / len(results)
    avg_neutral = sum([res['neutral'] for res in results]) / len(results)
    avg_negative = sum([res['negative'] for res in results]) / len(results)
    
    return avg_positive, avg_neutral, avg_negative


df = pd.read_csv("CEO_Letters_Dataset.csv", delimiter=";", encoding="latin1")  

output_data = []

for index, row in df.iterrows():
    company = row['company']
    year = row['year']
    letter = row['letter']

    print(f"Analyserar {company}, {year}...")

    results = analyze_text(letter)
    avg_pos, avg_neu, avg_neg = summarize_results(results)

    
    result_row = {
        "company": company,
        "year": year,
        "avg_positive": avg_pos,
        "avg_neutral": avg_neu,
        "avg_negative": avg_neg
    }

    for i, res in enumerate(results):
        result_row[f"chunk_{i+1}_pos"] = res["positive"]
        result_row[f"chunk_{i+1}_neu"] = res["neutral"]
        result_row[f"chunk_{i+1}_neg"] = res["negative"]

    output_data.append(result_row)

result_df = pd.DataFrame(output_data)
result_df.to_excel("sentiment_resultat.xlsx", index=False)

print("\nAnalys klar. Sparat i 'sentiment_resultat.xlsx'.")
