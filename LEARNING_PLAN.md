# Next steps before 16 October, 9:30 a.m. India time

1. Run the updated StudentLife app locally. Read the four actual feature definitions.
2. Reproduce preparation and training using the README commands.
3. Explain the baseline comparison honestly: the first model did not outperform the mean baseline on seven holdout participants.
4. Read the sleep category decoder and reverse-scoring tests. Explain why raw category numbers are not hours.
5. Check GLOBEM access eligibility through its official PhysioNet page. Until authorized files are available, describe its importer as implemented but unvalidated on real files.
6. Test local LLM generation if your computer supports Ollama; distinguish retrieval fallback from generation.
7. Publish code and aggregate results on GitHub. Exclude participant CSVs, RDS files, restricted data and virtual environments. Verify each staged file.
8. Record a 90-second demo covering problem, source, window, model, explanation, baseline and limitations.

Do not tune on the current seven-person holdout. A new experiment requires a documented evaluation design. Do not claim clinical accuracy, cloud deployment or GLOBEM evaluation without doing that work.

Step 1 EDA is implemented for the current four features. Open the Explore data tab. Next feature work can add activity and nighttime/variation summaries through a separate documented experiment; never tune using the prior holdout.

Step 2 candidate features are now implemented. Run engineer_studentlife.py and inspect the New features tab. Step 3 should compare the original and expanded features using development-only validation; no further tuning on the previously inspected holdout.

Step 3 is implemented. Run compare_models.py, then server.py --model-dir artifacts/studentlife_v2. Read Model evidence. Eleven features did not beat four in fixed CV, and nested selection did not beat the baseline. Next: Step 4, SHAP explanations with an explicit descriptive scope.

Step 4 is implemented. Install requirements-shap.txt, run add_shap.py and inspect signed contributions in Explore. Explain why attribution is not causation and why a feature can influence a weak model without making it useful. Next: privacy verification, followed by improving the opt-in nudge flow.

Step 5 is implemented. Run verify_privacy.py and inspect the Privacy tab. Explain the distinction between unsaved scenario inputs and persistent research data, and between an app contacting localhost and auditing the separate LLM service. Next: Step 6, a clearer opt-in nudge prototype.

Step 6 is implemented: choose a weekly focus in Explore, opt in, then request a suggestion. Explain TF-IDF retrieval versus optional LLM rewriting, the fallback, and why a weak predictive score must not select a support action. The six core steps are implemented; reproduce the demo locally and test live Ollama before claiming generation was demonstrated.
