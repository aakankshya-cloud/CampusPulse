# Dataset definitions and access

## StudentLife — implemented and executed

The public Zenodo archive is a conversion of Dartmouth StudentLife. The loneliness table has 46 pre and 37 complete post surveys. One row per participant is created; no survey score is copied into fabricated weekly labels. Questionnaire answers are never features.

Never/Rarely/Sometimes/Often map to 1/2/3/4. Reverse items 1, 4, 5, 6, 9, 10, 15, 16, 19, 20 using 5 minus response. Q4 explicitly says “I do not feel alone,” so it is positively worded and reversed. This follows all ten positive statements in the archive; some online templates omit Q4 from their printed reverse list despite that wording. Sum all 20 observed responses, dropping incomplete totals. The target is 20–80 without clinical cutoffs.

The fixed predictor window is 27 March 2013 inclusive through 24 April exclusive, America/New_York, intended to precede end-term measurement. The survey table has pre/post type but no individual completion timestamps: exact per-person temporal ordering is an unverified assumption.

| Feature | Definition | Limitation |
|---|---|---|
| Reported sleep | Median decoded EMA report per date, then mean across observed dates | Active self-report, not passive sensing |
| Sleep duration SD | Sample SD across those daily durations | Not bedtime variability |
| Conversation | Union overlapping intervals, clip to window, split at local midnight, average minutes across event-days | Not all social interaction; missing days are ambiguous |
| Phone locked | Same interval procedure; mean hours across event-days | Not screen time or all phone usage |

The EMA codebook maps sleep category 1 to <3 hours, 2 to 3.5, 3 to 4, through 19 to 12. Category 1 is left-censored and excluded rather than assigned invented hours. Undocumented code 0 and other invalid codes are excluded. This can bias sleep summaries; excluded counts are reported in metadata.

A day with no sensor event can reflect no event or missing recording. It is never filled with zero. Event-day means are conditional summaries with unknown complete recording coverage. At least seven observed sleep days and seven event-days in each sensor channel are required; 34 of 37 participants qualify. This eligibility rule can introduce selection bias. Local coverage.csv records counts.

Evaluation reserves seven students before five-fold grouped model selection on the other 27. Preprocessing stays inside each fold. Ridge was selected, then failed to beat the mean baseline on holdout. No tuning was performed against those results. With only seven test participants, precision and external validity are unestablished. No clinical validation, fairness audit or intervention evaluation has been performed. Future model development needs a new documented evaluation design or nested validation rather than repeated tuning on this holdout.

## GLOBEM — importer ready, real-file validation pending

Official access: https://physionet.org/content/globem/1.1/

The official resource requires credentialing, listed CITI training and a signed data-use agreement. Do not bypass these requirements or share credentials. Keep restricted records and derived artifacts locally as required by the agreement. This project includes none of those records.

GLOBEM documents a 10-item UCLA loneliness total on a 10–40 scale in pre/post surveys. Weekly depression benchmark labels are not loneliness labels. Do not multiply scores by two to treat the instruments as equivalent.

Once you have authorized files, inspect ONE cohort:

```powershell
.\.venv\Scripts\python.exe prepare_globem.py --cohort "D:\datasets\GLOBEM\INS-W_2" --inspect
```

The adapter expects FeatureData/rapids.csv and SurveyData/post.csv with pid/date keys, following the public documentation. Verify the exact scored UCLA post-total column and raw daily sensor column names. The script deliberately does not guess or score undocumented item formats. Confirm the chosen target is a total, not one questionnaire item.

Template only — replace every placeholder with a verified header:

```powershell
.\.venv\Scripts\python.exe prepare_globem.py --cohort "D:\datasets\GLOBEM\INS-W_2" --target-column "EXACT_UCLA_POST_TOTAL_COLUMN" --feature-columns "EXACT_RAW_FEATURE_COLUMN_1" "EXACT_RAW_FEATURE_COLUMN_2"
.\.venv\Scripts\python.exe train_real.py --prepared data/private_globem --output artifacts/globem
.\.venv\Scripts\python.exe server.py --model-dir artifacts/globem
```

Only raw f_slp, f_screen, f_steps, f_loc, f_blue or f_call features ending in :allday are accepted. Source units remain source units. Normalized (_norm), discretized (_dis) and history segments are excluded to avoid unreviewed participant-wide or future-information transforms.

For each post survey, average observed raw features over the preceding 28 days, excluding the survey date and future dates. Require seven observed days per selected feature. Reject duplicate participant/date rows or multiple post surveys. Date-shifted sensor and survey records must use the same participant-specific coordinate system; do not compare dates across participants as actual calendar dates.

The importer is tested on artificial fixtures only. It may require adaptation to your authorized files. It handles one cohort per run and does not perform cross-year validation. A later cross-year experiment must exclude repeat participants across train/test and audit source preprocessing. StudentLife and GLOBEM remain separate datasets and models; no transfer or combined-model claim is made.

Completed: StudentLife preparation and measured baseline comparison.
Pending authorized files: GLOBEM integration validation and evaluation.
Not established: reliable real-world loneliness inference, a combined model or successful cross-dataset generalization.

## Step 2 candidate definitions

See campuspulse/features.py and the Step 2 README table. The original four-feature experiment and eligible cohort remain the baseline. Candidate features are created for its 27 development participants only, preserving the same seven reserved IDs. Midnight–06:00 interval calculations use local New York dates and union overlapping intervals before counting. Half-window changes exclude dates outside the original 28-day window and do not fill unrecorded days with zeros. A negative change means a lower recorded average later, not an inferred psychological change. The new exercise channel decodes EMA have=1 as Yes and 2 as No; unknown or conflicting daily responses are excluded. The codebook for this field is in the public StudentLife R package EMA_definition.json.

## Step 3 validation policy

compare_models.py uses only the development-only feature CSV and locked split manifest from Step 2. It rejects any participant mismatch or overlap with reserved IDs. Fixed-candidate five-fold CV compares two feature sets with identical Ridge/random-forest settings plus a mean baseline. Five outer/four inner grouped folds separately estimate the selection procedure. No original reserved participant records are loaded by this command. Earlier EDA and model inspection mean the results are exploratory; no independent replication is claimed. The original model and its initial seven-person holdout result remain unchanged. The development-fitted updated demonstration model lives under artifacts/studentlife_v2.

## Step 4 explanation policy

SHAP explains the chosen development-fitted Ridge model using an aggregate development background reference. Independent/interventional linear attributions are reported, not correlation-dependent or causal effects. The global summary uses the same development data as the model and is descriptive. The seven reserved students are not used for background construction. See README for additivity checks, supported preprocessing, reference storage, dependency installation and retraining behavior.

## Step 5 privacy scope

The app does not store slider scenarios, but research downloads and prepared participant data are persistent local files. Keep those private and follow their source terms. Optional local-AI transport sends only topics and authored cards, with fixed destination, proxy bypass and redirect rejection. Its separate service is not audited. verify_privacy.py checks fictional scenarios and source fingerprints, not a phone, OS or legal compliance regime. Browser interaction testing of Clear scenario remains manual.

### Step 6: voluntary support focus
Choose a weekly support area independently of the research model inputs, then explicitly opt in. Offline authored retrieval requires no extra installation. The loaded real model does not infer a support need or choose an intervention. Optional local generation still requires separately installed Ollama and a model; this build was tested with mocked transport, not a live LLM.
