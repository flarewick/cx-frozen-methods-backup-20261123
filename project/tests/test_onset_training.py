import ast
import copy
import unittest
from cx_tta.paths import ROOT
from cx_tta.native_history_cache import read
from cx_tta.onset_train_contract import validate_protocol, SEEDS, STAGE


class IndependentTrainContract(unittest.TestCase):
    def setUp(self):
        self.p=read(ROOT/'outputs/reports/protocols/first_chunk_confirmation_20261006_151638_335660/protocol.json')
        self.p.update(stage=STAGE,seeds=SEEDS,confirmation_only=False,
            scene_split={str(s):'fit' if i<4 else 'calibration' for i,s in enumerate(SEEDS)})

    def test_no_confirmation_borrowing_or_automatic_approval(self):
        validate_protocol(self.p)
        for key,value in [('stage','first_chunk_whole_scene_confirmation'),('training_approved',True),
                          ('seeds',list(range(501200,501208))),('confirmation_only',True)]:
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError):validate_protocol(p)

    def test_split_and_candidate_cannot_change(self):
        p=copy.deepcopy(self.p);p['scene_split']['501304']='fit'
        with self.assertRaises(ValueError):validate_protocol(p)
        p=copy.deepcopy(self.p);p['candidate']['ARM_duration_queries']=2
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_actual_physical_rollout_matches_accepted_TRAIN_entry(self):
        def get(name):
            tree=ast.parse((ROOT/'cx_tta'/name).read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rollout'),include_attributes=False)
        self.assertEqual(get('collect_onset_training.py'),get('evaluate_arm_prescription_v2.py'))


if __name__=='__main__':unittest.main()
