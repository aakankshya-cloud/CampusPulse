# CampusPulse
[![Python checks](https://github.com/aakankshya-cloud/CampusPulse/actions/workflows/checks.yml/badge.svg)](https://github.com/aakankshya-cloud/CampusPulse/actions/workflows/checks.yml)
A Python research dashboard exploring relationships between student routine summaries and loneliness questionnaire scores.

- **ML:** real StudentLife data, participant-separated evaluation and model comparison.
- **Explainability:** SHAP contributions for the selected Ridge model.
- **AI:** opt-in suggestions using TF-IDF retrieval and optional local Ollama generation.
- **Engineering:** automated tests, HTTP APIs and GitHub Actions workflow.
- **Scope:** educational prototype; not a clinical assessment or reliable loneliness detector.

## What the app does

Users explore example routine values, view an estimated questionnaire score and inspect which inputs influence the model. Separately, they can choose a support topic and opt in to a suggestion.

The app does not collect phone data automatically. Real-data inputs represent reported sleep, sleep variability, detected conversation and phone-locked duration. Phone-locked duration is not screen time.

## Architecture

Data preparation → feature engineering → participant-separated model evaluation → prediction and SHAP → dashboard.

The support workflow is separate: explicit consent → topic selection → authored-card retrieval → optional local LLM → response checks or authored fallback.

## Measured results

| Experiment | Result |
|---|---|
| StudentLife cohort | 34 eligible students; 27 development and 7 original holdout |
| Original holdout MAE | Ridge 9.91 versus mean baseline 9.20 |
| Development-only nested selection MAE | 10.248 versus baseline 9.856 |
| Selected configuration | Ridge using the original four features |
| Windows automated tests | 61 passed |
| Latest live LLM smoke test | 12/12 responses accepted across four topics repeated three times |
| Median accepted LLM response time | 2.762 seconds |
| Retrieved source matches | 12/12 |

The model did not outperform the baseline. The original holdout was not reused for feature selection.

LLM acceptance measures successful generation and response checks, not quality or safety. Review found incomplete sentences and movement wording that assumed comfort/accessibility. Keyword checks are limited. Saved reports retain pending human-review fields; AI-assisted review is not human validation.

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

Choose a topic, tick consent and request a suggestion. The interface distinguishes local generation, offline retrieval and fallback. Without Ollama, authored offline suggestions remain available.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_privacy.py
```

With the StudentLife app running, evaluate the support endpoint:

```powershell
.\.venv\Scripts\python.exe evaluate_llm.py
```

Reports are saved under `artifacts/llm_evaluation`. These fixed-topic checks are a small smoke evaluation, not an independent wellbeing benchmark.

GitHub Actions is configured to run tests and synthetic training. A successful hosted run must be verified after pushing.

## Privacy and limitations

- Interactive scenarios are not intentionally persisted by the tested app handlers.
- Optional generation sends broad topics and authored cards to a fixed localhost endpoint.
- Separate Ollama service behavior, logs and downstream networking are not audited.
- Research files and explicitly generated evaluation reports persist locally.
- SHAP explains model behavior; it does not establish causation.
- The dataset is small and recording coverage is uneven.
- GLOBEM support is fixture-tested only; real restricted data was not obtained.
- Cloud deployment and clinical effectiveness have not been demonstrated.

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for research assumptions and [artifacts/VERIFICATION.md](artifacts/VERIFICATION.md) for development verification history.