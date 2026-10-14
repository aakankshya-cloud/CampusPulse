# CampusPulse

[![Python checks](https://github.com/aakankshya-cloud/CampusPulse/actions/workflows/checks.yml/badge.svg)](https://github.com/aakankshya-cloud/CampusPulse/actions/workflows/checks.yml)

A Python research dashboard exploring relationships between student routine summaries and loneliness questionnaire scores, with explainable ML and opt-in, source-based suggestions.

- **ML:** real StudentLife data, participant-separated evaluation and model comparison.
- **Explainability:** SHAP contributions for the selected Ridge model.
- **AI:** TF-IDF retrieval over 20 sourced cards and optional local Ollama selection of approved wording.
- **Engineering:** HTTP APIs, 69 passing Windows tests and a GitHub Actions workflow.
- **Scope:** educational research prototype; not a clinical assessment or reliable loneliness detector.

## What the app does

Users explore routine values, view an estimated questionnaire score and inspect which inputs influence the model. Separately, they can choose a support topic or enter a short search query and opt in to a suggestion.

The app does not collect phone data automatically. Real-data inputs represent reported sleep, sleep variability, detected conversation and phone-locked duration. Phone-locked duration is not screen time.

## Architecture

**Research workflow:** data preparation → feature engineering → participant-separated evaluation → prediction and SHAP → dashboard.

**Support workflow:** explicit consent → topic/query → local TF-IDF retrieval → optional local LLM selection → approved wording or authored fallback.

The support suggestion is separate from the predicted questionnaire score. Unrestricted LLM rewrites are not displayed.

## Measured results

| Experiment | Result |
|---|---|
| StudentLife cohort | 34 eligible students: 27 development and 7 original holdout |
| Original holdout MAE | Ridge 9.91 versus mean baseline 9.20 |
| Development-only nested selection MAE | 10.248 versus baseline 9.856 |
| Selected configuration | Ridge using the original four features |
| Windows automated tests | 69 passed |
| Retrieval evaluation | 20 hand-authored queries |
| Retrieval Hit@1 / Hit@3 | 90% / 100% |
| Retrieval MRR@3 | 0.950 |

Lower MAE is better. The model did not outperform the mean baseline. The original holdout was not reused for feature selection.

Retrieval results come from a small, project-authored regression set, not an independent user benchmark.

## Human review of the earlier LLM version

The earlier free-rewriting version accepted 12/12 responses across four topics repeated three times, with a median accepted-response time of 2.762 seconds. Acceptance alone did not establish response quality.

Project-author review, assisted by AI explanations of the rubric, found:

| Criterion | Responses rated Yes |
|---|---|
| Topic relevance | 12/12 |
| Faithfulness to the authored card | 9/12 |
| No added unsupported health claims | 12/12 |

Three movement responses failed to preserve comfort/accessibility conditions. Two routine responses also contained incomplete grammar.

These findings support the change to approved-wording selection. The completed review is saved separately from the unchanged original report:

[Completed human review](artifacts/llm_evaluation/human_review_20261014T195035895101Z.json)

This is a small, AI-assisted project-author review, not independent or clinical validation. It evaluates the historical rewriting version. Broader live evaluation of the updated selection version remains pending.

## Run a synthetic demonstration

Python 3.13 is the tested Windows setup. In PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-shap.txt
.\.venv\Scripts\python.exe train.py --demo
.\.venv\Scripts\python.exe server.py --model-dir artifacts
```

Open http://127.0.0.1:8000.

This demonstration uses fictional data. It does not reproduce StudentLife results.

## Reproduce the StudentLife workflow

Trained model files and participant records are excluded from Git. To create the real-data model locally:

```powershell
.\.venv\Scripts\python.exe prepare_studentlife.py
.\.venv\Scripts\python.exe explore_studentlife.py
.\.venv\Scripts\python.exe engineer_studentlife.py
.\.venv\Scripts\python.exe compare_models.py
.\.venv\Scripts\python.exe add_shap.py
.\.venv\Scripts\python.exe server.py --model-dir artifacts/studentlife_v2
```

Preparation requires internet access. See [REAL_DATA_GUIDE.md](REAL_DATA_GUIDE.md) for source definitions, timing assumptions and access limitations. Keep downloaded participant records local.

## Optional local LLM

Install Ollama separately, then download the model:

```powershell
ollama pull gemma3:1b
```

Stop CampusPulse and restart it from a terminal configured with:

```powershell
$env:OLLAMA_MODEL = "gemma3:1b"
.\.venv\Scripts\python.exe server.py --model-dir artifacts/studentlife_v2
```

Choose a topic or enter a short query, tick consent and request a suggestion.

The local LLM selects approved wording from retrieved cards. Unapproved output falls back to authored text. Without Ollama, offline authored suggestions remain available.

## Sourced support retrieval

The support feature searches 20 project-authored cards linked to NHS and Cornell guidance. Optional topic filtering narrows the search.

Cards are adaptations of general guidance, not quotations or evidence that CampusPulse improves wellbeing. Source links are displayed with suggestions.

## Verification

Run the automated tests, privacy checks and retrieval evaluation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_privacy.py
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

With the StudentLife app running, evaluate the support endpoint from a second terminal:

```powershell
.\.venv\Scripts\python.exe evaluate_llm.py
```

LLM reports are saved under `artifacts/llm_evaluation/`. These fixed-topic checks are a small smoke evaluation, not an independent wellbeing benchmark.

GitHub Actions runs automated tests and synthetic training. The badge above shows the workflow status.

## Privacy and limitations

- Interactive scenarios are not intentionally persisted by the tested app handlers.
- Support queries are searched locally; raw query text, identifiers, numeric inputs and predicted scores are omitted from the Ollama payload.
- Optional LLM requests send generalized topics and approved card wording to a fixed localhost endpoint.
- Separate Ollama service behavior, logs and downstream networking are not audited.
- Research files and explicitly generated evaluation reports persist locally.
- Approved-wording checks constrain displayed text; they do not establish clinical safety or effectiveness.
- SHAP explains model behavior, not causation.
- The dataset is small and recording coverage is uneven.
- GLOBEM support is fixture-tested only; real restricted data was not obtained.
- Cloud deployment and clinical effectiveness have not been demonstrated.

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for research assumptions and [artifacts/VERIFICATION.md](artifacts/VERIFICATION.md) for development verification history.