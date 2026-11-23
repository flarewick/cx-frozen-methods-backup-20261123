"""Declared source deltas relative to immutable ARM26; no policy math changes."""
CONFIGURE="    from native_discard_capture_v1.defer import configure\n    configure(task,branch=='M_event_full',item['case_id'])\n"
USAGE="     from native_unread_capture_full_v1.audit import record_usage\n     record_usage(task,branchfolder,branch,result)\n"
INSTALL="    from native_discard_capture_v1.defer import install as install_unread_capture\n    install_unread_capture()\n"
HOST_AUDIT="        from native_unread_capture_full_v1.audit import audit as audit_unread_capture\n        audit_unread_capture(root,spec)\n"
OLD_REFERENCE="   reference=item['task']not in reference_tasks and item['case_id']in getattr(schedule,'q0_pending',{})"
NEW_REFERENCE="   reference=True  # Whole original reference for EVERY fixed scene, independent of outcome."

def transform_runtime(text):
 assert text.count(OLD_REFERENCE)==1 and text.count('      if changed:')==1
 text=text.replace(OLD_REFERENCE,NEW_REFERENCE).replace('      if changed:','      if changed and not reference:')
 text=text.replace('from native_q0_task_handoff_v3.control import compare','from native_unread_capture_full_v1.audit import compare')
 assert text.count('    raw=lambda:c.call(lane,observe,task)')==1
 text=text.replace('    raw=lambda:c.call(lane,observe,task)',CONFIGURE+'    raw=lambda:c.call(lane,observe,task)')
 anchor="     write(branchfolder/'semantic_initial_receipt.json'"
 assert text.count(anchor)==1
 return text.replace(anchor,USAGE+anchor)

def restore_runtime(text):
 for marker in [CONFIGURE,USAGE,NEW_REFERENCE,'      if changed and not reference:']:assert text.count(marker)==1
 return text.replace(CONFIGURE,'').replace(USAGE,'').replace(NEW_REFERENCE,OLD_REFERENCE).replace('      if changed and not reference:','      if changed:').replace('from native_unread_capture_full_v1.audit import compare','from native_q0_task_handoff_v3.control import compare')
