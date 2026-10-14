"""Real SHAP LinearExplainer for the selected Ridge pipeline.

Interventional linear SHAP needs the background mean only. Store that aggregate,
not participant rows. Correlation-dependent attributions are not claimed.
"""
import hashlib
import json
import numpy as np
from sklearn.linear_model import Ridge


def signature(bundle):
    pipeline = bundle['model']
    state = {'features':bundle['report']['features']}
    for index, estimator in enumerate(pipeline.steps):
        name, fitted = estimator
        for key in ['statistics_','mean_','scale_','coef_','intercept_']:
            if hasattr(fitted,key):
                state[f'{index}:{name}:{key}'] = np.asarray(getattr(fitted,key)).tolist()
    return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()


def prepare(bundle, development):
    import shap
    if not isinstance(bundle['model'].steps[-1][1],Ridge):
        raise ValueError('This SHAP workflow supports the selected Ridge pipeline only. Do not substitute another explanation type silently.')
    names = bundle['report']['features']
    raw = development[names]
    transformed = bundle['model'][:-1].transform(raw)
    if transformed.shape[1] != len(names):
        raise ValueError('This explainer requires one transformed column per original input feature.')
    mean = np.asarray(transformed).mean(axis=0)
    # For an interventional linear model, this aggregate reference gives exactly
    # the same attributions as the full development background distribution.
    explainer = shap.LinearExplainer(bundle['model'].steps[-1][1],
                                     shap.maskers.Independent(mean.reshape(1,-1)))
    values = np.asarray(explainer.shap_values(transformed))
    base = float(explainer.expected_value)
    predictions = bundle['model'].predict(raw)
    errors = np.abs(base + values.sum(axis=1) - predictions)
    if not np.allclose(base + values.sum(axis=1),predictions,atol=1e-8,rtol=1e-8):
        raise ValueError('SHAP additivity check failed; no explanation will be saved.')
    bundle['shap_state'] = {'kind':'linear_interventional','features':names,
        'transformed_background_mean':mean.tolist(),'base_value':base,
        'model_signature':signature(bundle),'background_students':len(development)}
    labels = {s['id']:s['label'] for s in bundle['report']['feature_schema']}
    summary = {'method':'SHAP LinearExplainer with Independent masker',
        'scope':'Descriptive explanations of the development-fitted Ridge model, not new validation.',
        'reference':'Mean of transformed development features; equivalent to full development background for interventional linear SHAP.',
        'background_students':len(development), 'reserved_students_used':0,
        'base_value':base, 'max_additivity_error':float(errors.max()),
        'model_signature':bundle['shap_state']['model_signature'], 'shap_version':shap.__version__,
        'features':[{'feature':name,'label':labels[name],
                     'mean_absolute_shap':float(np.abs(values[:,i]).mean())}
                    for i,name in enumerate(names)],
        'limitations':['SHAP explains this fitted model; it does not improve its predictive validity.',
            'Interventional attribution breaks feature dependence; correlated inputs can receive different credit under other SHAP assumptions.',
            'These are score-point contributions, not probabilities, diagnoses or causal effects.',
            'Global summaries describe training/development behavior and are not independent performance evidence.',
            'Only aggregate background means are retained in the model artifact; no background participant rows are stored.']}
    bundle['report']['shap'] = summary
    return summary


def explain(bundle, row):
    state = bundle.get('shap_state')
    if state is None:
        return {'available':False,'reason':'SHAP has not been prepared for this model.'}
    if state['features'] != bundle['report']['features'] or state['model_signature'] != signature(bundle):
        return {'available':False,'reason':'The saved SHAP reference does not match the current model. Run add_shap.py again.'}
    try:
        import shap
    except ImportError:
        return {'available':False,'reason':'Install requirements-shap.txt to enable SHAP. Showing sensitivity checks instead.'}
    transformed = bundle['model'][:-1].transform(row)
    reference = np.asarray(state['transformed_background_mean']).reshape(1,-1)
    explainer = shap.LinearExplainer(bundle['model'].steps[-1][1],shap.maskers.Independent(reference))
    contributions = np.asarray(explainer.shap_values(transformed))[0]
    base = float(explainer.expected_value)
    prediction = float(bundle['model'].predict(row)[0])
    reconstruction = base + float(contributions.sum())
    if not np.isclose(reconstruction,prediction,atol=1e-8,rtol=1e-8):
        return {'available':False,'reason':'SHAP reconstruction failed. Showing sensitivity checks instead.'}
    labels={s['id']:s['label'] for s in bundle['report']['feature_schema']}
    effects=[{'feature':name,'label':labels[name], 'difference':float(value)}
             for name,value in zip(state['features'],contributions)]
    return {'available':True,'method':'Interventional linear SHAP',
        'base_value':base,'prediction':prediction,'reconstructed_prediction':reconstruction,
        'additivity_error':abs(reconstruction-prediction),
        'contributions':sorted(effects,key=lambda item:abs(item['difference']),reverse=True),
        'note':'Reference score + all contributions = prediction before rounding. Model explanation, not a causal claim.'}


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    rows=sorted(summary['features'],key=lambda item:item['mean_absolute_shap'])
    fig,axis=plt.subplots(figsize=(10,4.7),layout='constrained')
    axis.barh([r['label'] for r in rows],[r['mean_absolute_shap'] for r in rows],color='#237b54')
    axis.set_xlabel('Mean absolute SHAP contribution (questionnaire score points)')
    axis.set_title('What this model relies on across development students',loc='left',pad=18,fontweight='bold')
    axis.spines[['top','right']].set_visible(False)
    fig.supxlabel(f'{summary["background_students"]} development students · Fitted-model behavior, not accuracy, causation or an independent audit.',fontsize=10)
    fig.savefig(Path(output)/'global.png',dpi=160,facecolor='white');plt.close(fig)
