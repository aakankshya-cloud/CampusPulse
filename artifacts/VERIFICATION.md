# Verification — real-data update

- 56 unittest checks passed, including questionnaire polarity, incomplete targets, sleep category decoding, interval union/clipping, GLOBEM window exclusion, duplicate checks and scale validation.
- Fresh selective public-archive download executed; all four member CRCs and lengths verified.
- StudentLife preparation: 37 complete post surveys; 34 eligible participants.
- Training: 27 development / 7 holdout, zero student overlap. Ridge chosen by development CV.
- Holdout MAE 9.91 versus baseline 9.20; R² −0.20. No holdout improvement established and no tuning performed against it.
- Loaded StudentLife prediction and support HTTP endpoints verified.
- JavaScript syntax passed; visual browser checks unavailable due to failed browser download.
- GLOBEM tests use artificial fixtures only. No restricted records or trained GLOBEM model obtained.
- Live Ollama, Windows execution, Docker, GitHub-hosted CI and cloud deployment not verified here.

Step 1: development-only EDA generated and plots visually inspected. HTTP summary and PNG routes passed. A test confirms changing heldout feature values or scores cannot change descriptive feature statistics. JavaScript syntax passed. Model weights and holdout results were not changed.

Step 2: seven candidate features generated for 27 development students, seven reserved participants excluded. Exercise member download and integrity validation executed. Tests cover night-window overlap, half-window direction and minimum coverage, future-date exclusion, exercise response decoding/conflicts, and independence from reserved students. The candidate plot was visually inspected. No model retraining or new holdout evaluation occurred.

Step 3: five fixed configurations compared using 27 development students. Nested participant CV executed. Original four-feature Ridge retained; expanded features did not improve CV. Selection-procedure pooled MAE 10.248 versus baseline 9.856. No seven-person holdout re-evaluation performed. Tests verify fold participant separation, one outer prediction per student, reserved-ID rejection, and inference after model serialization. JavaScript syntax checked.

Step 4: SHAP 0.51.0 installed and LinearExplainer executed on the selected Ridge pipeline. Full-background and aggregate-reference attributions agree in tests. Maximum additivity error on the supplied development dataset: 0.0. Predictions before/after attachment are identical. Tests also cover missing-value preprocessing, stale-model signatures and honest unprepared fallback. Local SHAP API and PNG route verified; global plot visually inspected. No independent performance claim added.

Step 5: eleven scoped privacy checks passed and a code-fingerprinted report was generated. Prediction, prepared SHAP and offline retrieval were exercised with socket connection attempts prohibited. Consent/minimal payload checks and local transport guards passed. Tested inference/support handlers did not open files for writing via Python open APIs or emit request logs. Static browser-storage/default-consent checks, no-store/connection-policy headers and unexpected-Host rejection passed. Live Ollama downstream behavior, OS/browser-extension privacy and Clear scenario browser interaction remain outside verification.

Step 6: 56 tests passed in total. Four authored focus choices retrieved the expected cards; tests cover explicit consent, invalid focus, model-score independence, minimal generation payload and fallback preserving focus. Four real-model HTTP support requests passed. JavaScript syntax passed. Eleven privacy checks were rerun and their fingerprint matches this build. No live Ollama, browser interaction or clinical benefit claim is added.
