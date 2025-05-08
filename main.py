from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

candidate_labels = [
    "Muito Relevante",
    "Pouco Relevante",
    "Irrelevante ou Perigoso"
]

hypothesis_templates = {
    "pt": "Este termo de pesquisa está relacionado com o software de manutenção GP da Cegid para Portugal {}.",
    "es": "Este termo de pesquisa está relacionado con el software de mantenimiento GP de Cegid para España {}.",
    "latam": "Este termo de pesquisa está relacionado con el software de mantenimiento GP de Cegid para América Latina {}."
}

class Item(BaseModel):
    input: str = Field(..., min_length=1, description="Termo de pesquisa")
    region: str = Field(
        "pt",
        pattern="^(pt|es|latam)$",
        description="Mercado: 'pt', 'es' ou 'latam'"
    )

@app.post("/classify")
async def classify(item: Item):
    text = item.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input não pode ser vazio.")

    template = hypothesis_templates[item.region]
    try:
        result = classifier(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template=template
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no classificador: {e}")

    return {
        "search_term": text,
        "region": item.region,
        "hypothesis_template": template,
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4)
    }

