from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Carrega o modelo leve da Typeform
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

candidate_labels = [
    "Muito Relevante",
    "Pouco Relevante",
    "Irrelevante ou Perigoso"
]

class Item(BaseModel):
    input: str

@app.post("/classify")
async def classify(item: Item):
    if not item.input or not item.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    result = classifier(
        item.input,
        candidate_labels=candidate_labels,
        hypothesis_template="Este termo de pesquisa está relacionado com produtos do nosso portfólio como crédito habitação ou crédito pessoal Novo Banco {}."
    )

    return {
        "search_term": item.input,
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4)
    }
