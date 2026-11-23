import ast
import copy
import unittest
from cx_tta.paths import ROOT
from cx_tta.native_history_cache import read
from cx_tta.first_chunk_contract import validate_protocol, CONDITIONS, TASKS, SEEDS, STAGE


class ConfirmationContract(unittest.TestCase):
    def setUp(self):
        self.p = read(ROOT/'outputs/reports/protocols/arm_prescription_v2_20261006_134344_607783/protocol.json')
        self.p.update(stage=STAGE, conditions=list(CONDITIONS), tasks=list(TASKS), seeds=SEEDS,
            candidate=dict(ARM_duration_queries=1, ARM_strength=1, GRIP_strength=0,
                default_exit=True, threshold=.1, delay_queries=1, one_shot=True),
            confirmation_only=True, training_approved=False, no_new_training_or_router_or_TTA=True)

    def test_frozen_candidate_and_panel(self):
        validate_protocol(self.p)
        for key, value in [('ARM_duration_queries',2),('ARM_strength',.5),('GRIP_strength',1),
                           ('default_exit',False),('threshold',.2),('one_shot',False)]:
            p=copy.deepcopy(self.p); p['candidate'][key]=value
            with self.assertRaises(ValueError): validate_protocol(p)
        for key,value in [('stage','local_arm_prescription_TRAIN'),('training_approved',True),
                          ('conditions',['Z_zero','B_two_full']),('seeds',[501101,501102]),
                          ('compiled',False),('checkpoint_sha256','changed')]:
            p=copy.deepcopy(self.p); p[key]=value
            with self.assertRaises(ValueError): validate_protocol(p)

    def test_rollout_physics_unchanged_only_confirmation_cache_role(self):
        def rollout(name):
            module=ast.parse((ROOT/'cx_tta'/name).read_text())
            return next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='rollout')
        old=rollout('evaluate_arm_prescription_v2.py'); new=rollout('evaluate_first_chunk_confirmation.py')
        replacements=0
        for node in ast.walk(old):
            if isinstance(node,ast.keyword) and node.arg=='role':
                node.value=ast.Constant(value='diagnostic_confirmation'); replacements+=1
        self.assertEqual(replacements,1)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))


if __name__ == '__main__': unittest.main()
