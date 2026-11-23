import ast
import copy
import unittest
from cx_tta.paths import ROOT
from cx_tta.native_history_cache import read
from cx_tta.onset_support_contract import validate_protocol,SEEDS,STAGE


class SupportExtension(unittest.TestCase):
    def setUp(self):
        self.p=read(ROOT/'outputs/reports/protocols/onset_TRAIN_20261007_021608_309810/protocol.json')
        self.p.update(stage=STAGE,seeds=SEEDS,scene_split={str(s):'fit' if i<8 else 'calibration' for i,s in enumerate(SEEDS)},
            maximum_formal_rollouts=128,fixed_single_extension=True,outcome_based_stopping=False)

    def test_fixed_panel_split_candidate_and_stopping(self):
        validate_protocol(self.p)
        for key,value in [('stage','onset_paired_TRAIN'),('seeds',list(range(501300,501308))),
                          ('maximum_formal_rollouts',10000),('outcome_based_stopping',True),('no_scene_replacement',False)]:
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError):validate_protocol(p)
        for edit in ['split','candidate','coverage']:
            p=copy.deepcopy(self.p)
            if edit=='split':p['scene_split']['501408']='fit'
            elif edit=='candidate':p['candidate']['ARM_strength']=.5
            else:p['support_gate']['minimum_fit_negative_distinct_scenes']=1
            with self.assertRaises(ValueError):validate_protocol(p)

    def test_physical_collection_and_audit_equal_accepted_entry(self):
        def tree(file):return ast.dump(ast.parse((ROOT/'cx_tta'/file).read_text()),include_attributes=False)
        old=(ROOT/'cx_tta/collect_onset_training.py').read_text()
        expected=old.replace('.onset_train_contract','.onset_support_contract').replace('.onset_train_analysis','.onset_support_analysis').replace("group=('onset_training',args.task)","group=('onset_support',args.task)")
        self.assertEqual(tree('collect_onset_support.py'),ast.dump(ast.parse(expected),include_attributes=False))
        old=(ROOT/'cx_tta/onset_train_analysis.py').read_text().replace('.onset_train_contract','.onset_support_contract')
        self.assertEqual(tree('onset_support_analysis.py'),ast.dump(ast.parse(old),include_attributes=False))

    def test_screen_has_no_policy_outcomes_and_same_execution(self):
        old=(ROOT/'cx_tta/screen_onset_training.py').read_text().replace('.onset_train_contract','.onset_support_contract').replace("'onset_train_registries'","'onset_support_registries'").replace("'onset_TRAIN_eligibility_only'","'onset_support_eligibility_only'")
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse((ROOT/'cx_tta/screen_onset_support.py').read_text())))


if __name__=='__main__':unittest.main()
