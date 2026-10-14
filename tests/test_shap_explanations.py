import importlib.util
import unittest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from campuspulse.explain import prepare, explain

@unittest.skipUnless(importlib.util.find_spec('shap'), 'Optional SHAP dependency is not installed')
class ShapTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(9)
        self.data=pd.DataFrame(rng.normal(size=(27,3)),columns=['a','b','c'])
        self.data.loc[0,'b']=np.nan
        y=40+3*self.data.a-self.data.b.fillna(0)+rng.normal(0,.2,27)
        model=make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=2)).fit(self.data,y)
        self.bundle={'model':model,'report':{'features':['a','b','c'],
            'feature_schema':[{'id':c,'label':c} for c in self.data.columns]}}

    def test_aggregate_reference_equals_full_background_shap(self):
        import shap
        before=self.bundle['model'].predict(self.data)
        summary=prepare(self.bundle,self.data)
        np.testing.assert_array_equal(before,self.bundle['model'].predict(self.data))
        row=pd.DataFrame([{'a':1.2,'b':-.7,'c':.1}])
        result=explain(self.bundle,row)
        full=shap.LinearExplainer(self.bundle['model'].steps[-1][1],
            shap.maskers.Independent(self.bundle['model'][:-1].transform(self.data)))
        expected=full.shap_values(self.bundle['model'][:-1].transform(row))[0]
        by_name={c['feature']:c['difference'] for c in result['contributions']}
        np.testing.assert_allclose([by_name[c] for c in self.data.columns],expected,atol=1e-10)
        self.assertAlmostEqual(summary['base_value'],float(before.mean()),places=10)
        self.assertLess(result['additivity_error'],1e-8)
        self.assertEqual(len(self.bundle['shap_state']['transformed_background_mean']),3)

    def test_changed_model_invalidates_reference(self):
        prepare(self.bundle,self.data)
        self.bundle['model'].steps[-1][1].coef_[0]+=1
        result=explain(self.bundle,self.data.iloc[[1]])
        self.assertFalse(result['available'])
        self.assertIn('does not match',result['reason'])

    def test_unprepared_model_is_not_mislabeled_shap(self):
        result=explain(self.bundle,self.data.iloc[[1]])
        self.assertFalse(result['available'])

    def test_missing_input_is_explained_through_the_fitted_pipeline(self):
        prepare(self.bundle,self.data)
        result=explain(self.bundle,self.data.iloc[[0]])
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['reconstructed_prediction'],float(self.bundle['model'].predict(self.data.iloc[[0]])[0]),places=10)
