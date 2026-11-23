import ast
import copy
import json
from pathlib import Path
import unittest
from cx_tta.shared_expansion_contract import validate_protocol, SEEDS
from cx_tta.shared_onset_contract import validate_protocol as old_validate

ROOT=Path(__file__).resolve().parents[1]


class ExpansionContractTests(unittest.TestCase):
    def panel(self):
        parent=json.loads((ROOT/'outputs/reports/protocols/shared_onset_confirmation_20261007_103729_317776/protocol.json').read_text())
        old_validate(parent)
        panel=copy.deepcopy(parent)
        panel.update(stage='shared_onset_expansion',seeds=SEEDS,
                     scene_split={str(s):'confirmation' for s in SEEDS},
                     maximum_formal_rollouts=512,process_wall_time_limit_s=21600)
        return parent,panel

    def test_old_and_new_panels_are_separate(self):
        old,new=self.panel();validate_protocol(new)
        with self.assertRaises(ValueError):validate_protocol(old)
        with self.assertRaises(ValueError):old_validate(new)
        self.assertFalse(set(old['seeds']) & set(new['seeds']))

    def test_reject_scope_or_gate_changes(self):
        _,p=self.panel()
        edits=[('seeds',p['seeds'][:-1]),('maximum_formal_rollouts',1024),
               ('process_wall_time_limit_s',43200),('training_approved',True),
               ('gate_rule',dict(harm_cost=3,threshold=0,calibrated=False,retry=False)),
               ('checkpoint_sha256','bad'),('process_blocks',[0]),
               ('internal_noise',False),('per_action_observations',False)]
        for key,value in edits:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):validate_protocol(q)

    def test_original_rollout_and_audit_body_unchanged(self):
        for old,new,name in [('collect_shared_onset','collect_shared_expansion','rollout'),
                             ('shared_onset_analysis','shared_expansion_analysis','summarize')]:
            def body(module):
                tree=ast.parse((ROOT/'cx_tta'/f'{module}.py').read_text())
                return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
            self.assertEqual(body(old),body(new))


if __name__=='__main__':unittest.main()
