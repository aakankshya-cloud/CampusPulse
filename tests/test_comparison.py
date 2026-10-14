import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from campuspulse.compare import compare
from campuspulse.model import score
from campuspulse.studentlife import SCHEMA

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        self.df=pd.DataFrame({'student_id':[f'fixture_{i:02d}' for i in range(20)]})
        for item in SCHEMA:
            self.df[item['id']]=rng.uniform(item['min'],item['max'],20)
        self.df['fixture_extra']=rng.uniform(0,1,20)
        self.df.loc[0,'fixture_extra']=np.nan
        self.df['loneliness_score']=rng.uniform(25,65,20)
        self.metadata={'development_only':True,'target':'Fixture outcome',
            'target_bounds':[20,80],'limitations':[],
            'features':SCHEMA+[{'id':'fixture_extra','label':'Fixture extra','unit':'fraction',
                               'min':0,'max':1,'step':.01}]}
        self.split={'development_ids':self.df.student_id.tolist(),
                    'reserved_ids':['reserved_fixture'],'original_dataset_sha256':'fixture-fingerprint'}

    def test_nested_evaluation_and_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            report,private=compare(self.df,self.metadata,self.split,directory,outer_splits=3,inner_splits=2,trees=5)
            self.assertNotIn('holdout',report)
            self.assertEqual(report['reserved_students'],1)
            self.assertEqual(len(report['comparison']),5)
            self.assertEqual(len(report['nested_folds']),3)
            validation_ids=[]
            for fold in private['folds']:
                self.assertFalse(set(fold['train_ids'])&set(fold['validation_ids']))
                validation_ids.extend(fold['validation_ids'])
                self.assertNotIn('reserved_fixture',fold['train_ids']+fold['validation_ids'])
            self.assertEqual(sorted(validation_ids),sorted(self.df.student_id))
            bundle=joblib.load(directory+'/model.joblib')
            values={f['id']:f['default'] for f in report['feature_schema']}
            self.assertTrue(np.isfinite(score(bundle,values)['estimate']))

    def test_reserved_person_is_rejected_before_fitting(self):
        self.df.loc[0,'student_id']='reserved_fixture'
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError,'locked development'):
            compare(self.df,self.metadata,self.split,directory,trees=5)
