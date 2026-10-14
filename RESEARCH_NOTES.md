# Three project directions, assessed for this deadline

Research checked 14 October 2026. These are judgments about scope and learning value, not an objective ranking of all projects online. The full internship posting was unavailable; fit is inferred from the requirements quoted in the conversation.

| Project | ML work | AI work | My assessment |
|---|---|---|---|
| **CampusPulse: student behavior and wellbeing research prototype** | Behavioral regression, participant-grouped evaluation, missingness and explainability | Retrieval-grounded optional support messages | Continue this because you explicitly selected it. A working synthetic demo is achievable quickly; real questionnaire timing and small-sample validation remain the main risks. |
| **ReadmissionLens: hospital readmission prediction and evidence dashboard** | Predict 30-day readmission; split by patient; address imbalance, PR-AUC and calibration | Summarize model factors without issuing care instructions | Strongest alternative for structured healthcare DS. UCI's historical diabetes dataset provides a defined outcome, but careful leakage review and honest age/generalization limits are essential. |
| **CareFlow: synthetic patient-record quality and retrieval assistant** | Detect injected data-quality anomalies with a documented evaluation set | Retrieve record sections and generate source-linked summaries | Strong for AI + healthcare engineering and learning FHIR. ML anomaly labels would be constructed, so do not claim real fraud detection or clinical validation. |

## Primary sources

- [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008): describes historical clinical encounters and the early-readmission target.
- [Synthea](https://synthetichealth.github.io/synthea/): official synthetic-patient generator; supports healthcare records in FHIR and other formats. Synthetic records do not establish clinical model performance.
- [Dartmouth StudentLife](https://studentlife.cs.dartmouth.edu/): candidate original research source. The original site timed out; the Zenodo conversion was subsequently retrieved, integrity checked and integrated. See REAL_DATA_GUIDE.md for measured results.
- [MIT MIMIC access guidance](https://mimic.mit.edu/docs/faq/how-to-get-access.html): credentialing, training and a data-use agreement make it a poor dependency for this short deadline. Do not seek copies that bypass access requirements.
- [Ollama chat API](https://docs.ollama.com/api/chat): the local optional generation integration follows the documented chat endpoint and non-streaming request shape.

## Choice

Proceed with CampusPulse as requested. Its strength will depend on evaluation, reproducibility, understanding, and honest documentation. A voice feature alone will not compensate for missing data evidence.
