from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI()

# Carrega o modelo leve da Typeform
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

# Labels de relevância
candidate_labels = [
    "Muito Relevante",
    "Pouco Relevante",
    "Irrelevante ou Perigoso"
]

# Hypothesis templates por mercado
hypothesis_templates = {
    "pt": "Este termo de pesquisa está relacionado com o software de manutenção GP da Cegid para Portugal.",
    "es": "Este termo de pesquisa está relacionado con el software de mantenimiento GP de Cegid para España.",
    "latam": "Este termo de pesquisa está relacionado con el software de mantenimiento GP de Cegid para América Latina."
}

class Item(BaseModel):
    input: str = Field(..., min_length=1, description="Termo de pesquisa")
    region: str = Field(
        "pt",
        regex="^(pt|es|latam)$",
        description="Mercado: 'pt', 'es' ou 'latam'"
    )

@app.post("/classify")
async def classify(item: Item):
    # valida input
    text = item.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    # escolhe o template consoante a região
    template = hypothesis_templates.get(item.region, hypothesis_templates["pt"])

    # invoca o classifier
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=template
    )

    return {
        "search_term": text,
        "region": item.region,
        "hypothesis_template": template,
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4)
    }
