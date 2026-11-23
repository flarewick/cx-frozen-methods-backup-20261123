"""Exercise fixed confirmation parser/models on six already-audited old cases only."""
from pathlib import Path
import numpy as np
from cx_tta_feedback_confirm5.analysis import evaluate
from cx_tta_feedback_confirm5.paths import TTA,OUTPUT,read,digest,write
def run():
 file=TTA/'outputs/native_online_bridge_recovery_v3/whole_case_audit.json';recovered=read(file)
 assert recovered['passed']and recovered['whole_cases_verified']==6
 records=[dict(case_id=r['case_id'],folder=r['folder'],available=True)for r in recovered['rows']]
 report,arrays,inputs=evaluate(records,independent=False,expected_cases=6)
 assert report['complete']and not report['independent_feedback_gate_passed']
 assert report['prequery_prediction_exact_checks']>0 and report['Z_sham_control_checks']>0
 path=OUTPUT/'developer_parser_predictions.npz'
 with path.open('xb')as f:np.savez_compressed(f,**arrays)
 report.update(parser_software_contract_passed=True,old_developer_not_independent=True,
   source_recovery_sha256=digest(file),predictions_sha256=digest(path),input_sha256=inputs,new_independent_scenes=0)
 write(OUTPUT/'developer_parser_replay.json',report)
 print('FIXED_CONFIRMATION_OLD_DATA_PARSER_PASS',report['all_rows'],report['prequery_prediction_exact_checks'],report['Z_sham_control_checks'],flush=True)
if __name__=='__main__':run()
