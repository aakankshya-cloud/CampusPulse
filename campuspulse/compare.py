"""Participant-grouped nested CV on development data, with no reserved outcomes."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .contracts import validate_real_training
from .studentlife import SCHEMA


def specifications(all_features, trees=120):
    base = [s['id'] for s in SCHEMA]
    def ridge():
        return make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=10))
    def forest():
        return make_pipeline(SimpleImputer(keep_empty_features=True), RandomForestRegressor(
            n_estimators=trees, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=1))
    return {
        'Mean baseline': (base, make_pipeline(SimpleImputer(keep_empty_features=True), DummyRegressor())),
        'Ridge · original 4': (base, ridge()),
        'Forest · original 4': (base, forest()),
        f'Ridge · expanded {len(all_features)}': (all_features, ridge()),
        f'Forest · expanded {len(all_features)}': (all_features, forest())}


def compare(df, metadata, split, output, outer_splits=5, inner_splits=4, trees=120):
    if not metadata.get('development_only'):
        raise ValueError('Use the development-only output from engineer_studentlife.py.')
    if df.student_id.duplicated().any():
        raise ValueError('Expected one row per development student.')
    ids = set(df.student_id.astype(str))
    if ids != set(split['development_ids']) or ids & set(split['reserved_ids']):
        raise ValueError('Participant membership differs from the locked development split.')
    df = validate_real_training(df.copy(), metadata['features'], metadata['target_bounds'])
    names = [f['id'] for f in metadata['features']]
    if not set(f['id'] for f in SCHEMA).issubset(names):
        raise ValueError('Original four features are required for a fair comparison.')
    specs = specifications(names, trees)
    groups, target = df.student_id, df.loneliness_score
    outer = list(GroupKFold(outer_splits).split(df, target, groups))
    fixed_predictions = {name:np.full(len(df), np.nan) for name in specs}
    fixed_folds = {name:[] for name in specs}
    fitted = {name:[] for name in specs}
    nested_predictions = np.full(len(df), np.nan)
    nested_folds, membership = [], []
    for fold, (train, test) in enumerate(outer, 1):
        training, validation = df.iloc[train], df.iloc[test]
        assert not set(training.student_id) & set(validation.student_id)
        inner_errors = {}
        for name, (columns, estimator) in specs.items():
            errors = -cross_val_score(clone(estimator), training[columns], training.loneliness_score,
                groups=training.student_id, cv=GroupKFold(inner_splits), scoring='neg_mean_absolute_error')
            inner_errors[name] = float(errors.mean())
            model = clone(estimator).fit(training[columns], training.loneliness_score)
            predictions = model.predict(validation[columns])
            fixed_predictions[name][test] = predictions
            fixed_folds[name].append(float(mean_absolute_error(validation.loneliness_score, predictions)))
            fitted[name].append(model)
        chosen = min(inner_errors, key=inner_errors.get)
        nested_predictions[test] = fixed_predictions[chosen][test]
        nested_folds.append({'fold':fold, 'selected_by_inner_cv':chosen,
            'mae':float(mean_absolute_error(validation.loneliness_score,nested_predictions[test])),
            'train_students':len(train),'validation_students':len(test), 'participant_overlap':0})
        # Participant names remain private; exported report contains counts only.
        membership.append({'fold':fold, 'train_ids':training.student_id.astype(str).tolist(),
                            'validation_ids':validation.student_id.astype(str).tolist()})
    assert np.isfinite(nested_predictions).all()
    rows = [{'model':name,'cv_mae':float(np.mean(fixed_folds[name])),
             'cv_std':float(np.std(fixed_folds[name])),
             'pooled_oof_mae':float(mean_absolute_error(target,fixed_predictions[name])),
             'features':len(specs[name][0]), 'fold_mae':fixed_folds[name]} for name in specs]
    winner = min(rows, key=lambda r:r['cv_mae'])['model']
    selected_columns, selected_estimator = specs[winner]
    # This is a development-fitted demonstration model, not independent test validation.
    final_model = clone(selected_estimator).fit(df[selected_columns], target)
    selected_schema = [{**s, 'default':float(df[s['id']].median())}
                       for s in metadata['features'] if s['id'] in selected_columns]
    # Match input column order exactly, regardless of metadata ordering.
    schema_map = {s['id']:s for s in selected_schema}
    selected_schema = [schema_map[c] for c in selected_columns]
    importance_vectors = []
    for (train, test), model in zip(outer, fitted[winner]):
        importance_vectors.append(permutation_importance(model,df.iloc[test][selected_columns],
            target.iloc[test],scoring='neg_mean_absolute_error',n_repeats=8,random_state=42).importances)
    importance = np.concatenate(importance_vectors, axis=1)
    baseline_mae = float(mean_absolute_error(target,fixed_predictions['Mean baseline']))
    nested_mae = float(mean_absolute_error(target,nested_predictions))
    report = {
        'source':'studentlife_nested_cv', 'evaluation_kind':'nested_cv_development_only',
        'warning':'Development-only research model. No new independent holdout evaluation or clinical validation.',
        'target_description':metadata['target'], 'selected_model':winner,
        'features':selected_columns,'feature_schema':selected_schema,'scale':metadata['target_bounds'],
        'comparison':rows, 'train_students':len(df),'test_students':0,
        'reserved_students':len(split['reserved_ids']), 'student_overlap':0,
        'validation':{'mae':nested_mae,
            'rmse':float(np.sqrt(mean_squared_error(target,nested_predictions))),
            'r2':float(r2_score(target,nested_predictions)), 'baseline_mae':baseline_mae},
        'nested_folds':nested_folds,
        'selection': f'Lowest mean {outer_splits}-fold development CV MAE; nested CV separately estimates the selection procedure.',
        'evaluation':f'{outer_splits} outer participant folds; {inner_splits} inner folds for selection. All preprocessing fitted inside folds.',
        'importance':[{'feature':s['id'],'label':s['label'],'mae_increase':float(v),'std':float(sd)}
            for s,v,sd in zip(selected_schema,importance.mean(axis=1),importance.std(axis=1))],
        'importance_method':'Out-of-fold permutation importance for the development-selected configuration; exploratory, not an independent attribution audit.',
        'dataset_sha256':hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
        'original_dataset_sha256':split['original_dataset_sha256'],
        'beats_baseline_in_nested_cv':nested_mae < baseline_mae,
        'provenance':metadata,
        'limitations':metadata['limitations']+[
            'Nested CV uses only 27 development students; the original seven held-out people were not accessed.',
            'Development-only CV cannot replace external validation. Prior exploratory inspection makes these results provisional.',
            'Fixed-candidate CV results support selection, not a separate confirmation of the selected winner.',
            'Nested CV estimates the selection procedure, not an independent error measurement for the final fitted model.',
            'Feature expansion is compared at the same fixed estimator settings; no hyperparameter sweep or outcome threshold is introduced.']}
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    medians = df[selected_columns].median().to_dict()
    joblib.dump({'model':final_model,'medians':medians,'report':report},output/'model.joblib')
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    # Keep split membership and participant-level predictions out of public artifacts.
    private = {'folds':membership, 'student_ids':df.student_id.astype(str).tolist(),
               'target':target.tolist(),'nested_predictions':nested_predictions.tolist()}
    return report, private
