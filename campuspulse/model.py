import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES, TARGET, LABELS, BOUNDS, validate_input, validate_training
from .contracts import validate_schema_input, validate_real_training


def partition(df):
    """Never put the same student's observations in both partitions."""
    return next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                .split(df, groups=df.student_id))


def train(df, output, source='synthetic', target_description='Invented demo score',
          schema=None, target_bounds=(20, 80), provenance=None):
    df = validate_real_training(df.copy(), schema, target_bounds) if schema else validate_training(df.copy())
    features = [s['id'] for s in schema] if schema else FEATURES
    labels = {s['id']: s['label'] for s in schema} if schema else LABELS
    train_idx, test_idx = partition(df)
    development, heldout = df.iloc[train_idx], df.iloc[test_idx]
    models = {
        'Mean baseline': make_pipeline(SimpleImputer(), DummyRegressor()),
        'Ridge regression': make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10)),
        'Random forest': make_pipeline(SimpleImputer(), RandomForestRegressor(
            n_estimators=120, min_samples_leaf=8, max_depth=8, random_state=42, n_jobs=1))}
    comparison = []
    for name, estimator in models.items():
        errors = -cross_val_score(estimator, development[features], development[TARGET],
            groups=development.student_id, cv=GroupKFold(5), scoring='neg_mean_absolute_error')
        comparison.append({'model': name, 'cv_mae': float(errors.mean()),
                           'cv_std': float(errors.std())})
    winner = min(comparison, key=lambda item: item['cv_mae'])['model']
    model = models[winner].fit(development[features], development[TARGET])
    prediction = model.predict(heldout[features])
    baseline = models['Mean baseline'].fit(development[features], development[TARGET])
    baseline_mae = mean_absolute_error(heldout[TARGET], baseline.predict(heldout[features]))
    importance = permutation_importance(model, heldout[features], heldout[TARGET],
        scoring='neg_mean_absolute_error', n_repeats=8, random_state=42)
    report = {
        'source': source, 'target_description': target_description,
        'warning': 'Synthetic scores demonstrate software only. No real-world validity.'
            if source == 'synthetic' else 'Research prototype. External validity is unestablished.',
        'selected_model': winner, 'comparison': comparison,
        'holdout': {'mae': float(mean_absolute_error(heldout[TARGET], prediction)),
                    'rmse': float(np.sqrt(mean_squared_error(heldout[TARGET], prediction))),
                    'r2': float(r2_score(heldout[TARGET], prediction)),
                    'baseline_mae': float(baseline_mae)},
        'train_students': int(development.student_id.nunique()),
        'test_students': int(heldout.student_id.nunique()),
        'train_rows': len(development), 'test_rows': len(heldout),
        'student_overlap': len(set(development.student_id) & set(heldout.student_id)),
        'features': features,
        'importance': [{'feature': f, 'label': labels[f], 'mae_increase': float(v),
                        'std': float(s)} for f, v, s in zip(features,
                            importance.importances_mean, importance.importances_std)],
        'dataset_sha256': hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
        'evaluation': '5-fold participant-grouped CV for selection, separate 20% student holdout.',
        'limitations': ['Observational associations are not causal.',
            'No diagnostic thresholds, clinical accuracy, or treatment effectiveness established.',
            'No demographic fairness audit: sensitive attributes were not supplied.',
            'Generalization to other campuses and future weeks is untested.',
            'The holdout evaluates unseen students, not forecasting future weeks.']}
    report['scale'] = list(target_bounds)
    report['feature_schema'] = [dict(item, default=float(development[item['id']].median())) for item in schema] if schema else None
    report['provenance'] = provenance
    if provenance:
        report['limitations'] += provenance.get('limitations', [])
    report['beats_baseline_on_holdout'] = report['holdout']['mae'] < baseline_mae
    bundle = {'model': model, 'medians': development[features].median().to_dict(), 'report': report}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / 'model.joblib')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def score(bundle, payload):
    schema = bundle['report'].get('feature_schema')
    features = bundle['report']['features']
    labels = {s['id']: s['label'] for s in schema} if schema else LABELS
    values = validate_schema_input(payload, schema) if schema else validate_input(payload)
    row = pd.DataFrame([values], columns=features)
    estimate = float(bundle['model'].predict(row)[0])
    effects = []
    for feature in features:
        changed = row.copy()
        changed[feature] = bundle['medians'][feature]
        reference = float(bundle['model'].predict(changed)[0])
        effects.append({'feature': feature, 'label': labels[feature],
                        'difference': round(estimate - reference, 2),
                        'reference': round(bundle['medians'][feature], 2)})
    from .explain import explain
    shap_result = explain(bundle, row)
    result = {'estimate': round(estimate, 1), 'source': bundle['report']['source'],
            'explanations': sorted(effects, key=lambda item: abs(item['difference']), reverse=True),
            'explanation_method': 'Change one input to its training median; not SHAP or causal effects.',
            'warning': bundle['report']['warning'], 'shap': shap_result}
    if shap_result['available']:
        result['explanations'] = shap_result['contributions']
        result['explanation_method'] = shap_result['method']
    return result
