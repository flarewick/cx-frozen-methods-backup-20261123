CAMERA="    from native_used_cameras_v1.camera import configure as configure_used_camera\n    configure_used_camera(task,branch=='M_event_full')\n"
SOURCE="""      if branch=='M_event_full'and selected_source is ObjectSource:
       from native_object_state_storage_v2.core import CPUStateSource
       selected_source=CPUStateSource
      from native_storage_capture_v1.source import tracked
      selected_source=tracked(selected_source,branchfolder)
"""
INSTALL="    from native_used_cameras_v1.camera import install as install_used_camera\n    install_used_camera()\n"
HOST="        from native_storage_capture_v1.audit import audit as audit_storage\n        audit_storage(root,spec)\n"
def transform(text):
 anchor="    configure(task,branch=='M_event_full',item['case_id'])\n";assert text.count(anchor)==1;text=text.replace(anchor,anchor+CAMERA)
 anchor='      return selected_source(';assert text.count(anchor)==1;text=text.replace(anchor,SOURCE+anchor)
 return text.replace('from native_unread_capture_full_v1.audit import compare','from native_storage_capture_v1.audit import compare')
def restore(text):
 assert text.count(CAMERA)==1 and text.count(SOURCE)==1
 return text.replace(CAMERA,'').replace(SOURCE,'').replace('from native_storage_capture_v1.audit import compare','from native_unread_capture_full_v1.audit import compare')
