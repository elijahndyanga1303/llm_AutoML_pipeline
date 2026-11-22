# AutoML system: n8n + FastAPI + LangGraph + LangChain

I built this to upload a CSV, pick an algorithm and the EDA plots I want, and get a trained model
and a report back without writing code. An LLM reads a statistical profile of the data (never the
raw rows) and decides the target column, which columns to drop and which columns to plot. My own
code draws the plots and trains the model, and n8n emails me the report.

## How it works

1. An n8n form takes the CSV, the algorithm, the plot types and an optional target column.
2. n8n sends them to a FastAPI endpoint, which starts a LangGraph pipeline in the background.
3. The pipeline profiles the data, asks the LLM to analyse it, validates the answer (retrying if
   it's invalid), renders the plots, trains the model and writes the report.
4. n8n checks the run status every 15 seconds and emails the report when it's done.

## Design choices

- The LLM only sees column statistics and a few sample values. It's cheaper, faster and more
  private, and it stays within free-tier limits.
- The LLM returns JSON plot specs, not code. Only whitelisted functions draw anything.
- Everything the LLM returns is validated against the real column names.
- Training is plain Python. The LLM only influences the target, task type and dropped columns.

## Run it

```bash
cp .env.example .env
docker compose up --build
```

I use Groq because the API key is free (https://console.groq.com/keys). In `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=
```

After changing `.env`, I run `docker compose up -d --force-recreate api`.
The API is at http://localhost:8000/docs and n8n is at http://localhost:5678. If a port is
taken, I change the left number in the `ports` line of `docker-compose.yml`.

To test the API without n8n (on Windows, as one line, using `curl.exe` in PowerShell):

```bash
curl -F "file=@sample_data/customers.csv" -F "algo=random_forest" -F "plots=hist,count,box,scatter,bar,heatmap_corr" http://localhost:8000/runs
curl http://localhost:8000/runs/<id>
```

Then I open `http://localhost:8000/runs/<id>/report` in the browser.

## n8n workflow

I import `n8n/automl_workflow.json` (Workflows → Import from file), add an SMTP credential to the
two email nodes (an app password for Gmail), and turn the workflow on. The form's Production URL
is in the Upload form node. If n8n runs outside this Docker setup, I change `http://api:8000` in
the three HTTP nodes to `http://localhost:8000`.

Things to remember:

- The uploaded file's binary name varies by n8n version, so the workflow uses
  `{{ Object.keys($binary)[0] }}` to find it.
- Test mode uses the Test URL. Shared links need the workflow to be active and the Production URL.
- If I set `API_KEY` in `.env`, I add an `X-API-Key` header to the three HTTP nodes.

## API

- `POST /runs`: upload a file with `algo`, `plots` and an optional `target`
- `GET /runs/{id}`: status, current stage and results
- `GET /runs/{id}/report`: HTML report with the plots embedded
- `GET /runs/{id}/model`: the trained `model.joblib`



## Limits and next steps

CSV only, default hyperparameters, and datetime columns are dropped. Next I'd add a job queue with
Postgres, MLflow tracking and a confirmation step for the target column.