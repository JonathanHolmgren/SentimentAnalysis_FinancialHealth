import pandas as pd
import openai
from openai import OpenAI


# API-Key
client = OpenAI(api_key="")

#CSV-filen 
df = pd.read_csv("CEO_Letters_Dataset.csv", delimiter=";", encoding="latin1")  #"ISO-8859-1"


def process_letter(letter_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {
                    "role": "system",
                     "content": (
                        "Du får ett textutdrag som kommer från avsnittet 'VD:s ord' i en årsredovisning. "
                        "Analysera endast innehållet i detta utdrag, bortse från all annan information. "
                        "Din uppgift är att göra en sentimentanalys i tre kategorier: positivt, neutralt och negativt. "
                        "Varje kategori ska ha ett värde mellan 0.0 och 1.0, och summan ska vara exakt 1.0. "
                        "Svar ska alltid ges i följande format:\n\n"
                        "Poäng: 0.75 positivt, 0.15 neutralt, 0.10 negativt\n\n"
                        "Motivering: [Här kommer en kort förklaring till varför dessa poäng har satts baserat på textens ton och innehåll.]"
    ) },
                {
                    "role": "user",
                    "content": letter_text
                }
            ],
            
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API request error: {e}"


df["response"] = df["letter"].apply(process_letter)


df.to_excel("sentiment_resultat.xlsx", index=False)

print("Klar! Resultatet är sparat i sentiment_resultat.xlsx'.")