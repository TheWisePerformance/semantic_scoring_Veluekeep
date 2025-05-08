# Usa imagem base pequena do Python
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Copiar requirements.txt e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY main.py .

# Expor a porta 8000
EXPOSE 8000

# Comando para arrancar a aplicação
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}

