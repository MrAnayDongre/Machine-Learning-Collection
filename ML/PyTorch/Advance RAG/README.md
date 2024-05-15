# Run RAG locally

## Install Ollama

[https://ollama.com](https://ollama.com)

## For Development

### Setup PostgreSQL with PgVector in Docker

You can change postgresql environment variables in the `.env` file

```bash
docker compose -f docker-compose.dev.yml up -d
```

### Install python dependencies

```bash
pip install -r requirements.txt
```

Also you can use `pipenv`

```bash
pipenv install
```

### Look into the notebook

[Local RAG Notebook](notebook/local-rag.ipynb)

### Look into the API

[Local RAG API](api)

### Run the API

```bash
python server.py
```

or

```bash
uvicorn api:app --reload
```

## For "Production"

```bash
docker compose up
```

or

```bash
docker compose up -d
```

I recommend running `docker compose up` to see the logs and check if everything is working.

The api container will try to connect to ollama server and pull the necessary models, this can take a while.

Feel free to change the `docker-compose.yml` and `.env` file to use your own models and configurations.

---
If you found this project helpful and would like to show your support, please consider giving it a ⭐️ on GitHub! Your star will help others discover this repository and contribute to its growth. 🌟
