"""Install only after a real Python SIGINT handler; original handler remains authoritative."""
from pathlib import Path
import ctypes,os,signal,struct
LIBRARY=Path(__file__).resolve().parent/'libreceipt.so'
FORMAT=struct.Struct('<iiiIqqqQQQQQ')

class Hook:
 def __init__(self,root):
  root=Path(root);handler=signal.getsignal(signal.SIGINT);assert callable(handler)
  self.lib=ctypes.CDLL(str(LIBRARY));self.lib.install_receipt.argtypes=[ctypes.c_int];self.lib.install_receipt.restype=ctypes.c_long
  self.lib.restore_receipt.restype=ctypes.c_long;self.lib.receipt_record_size.restype=ctypes.c_long
  assert self.lib.receipt_record_size()==FORMAT.size
  self.fd=os.open(root/'SIGINT_kernel_receipts.bin',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_CLOEXEC,0o600)
  try:assert self.lib.install_receipt(self.fd)==0
  except BaseException:os.close(self.fd);raise
  assert signal.getsignal(signal.SIGINT)is handler
 def close(self):
  if self.fd is not None:
   assert self.lib.restore_receipt()==0;os.close(self.fd);self.fd=None

def read_receipts(file):
 data=Path(file).read_bytes();assert len(data)%FORMAT.size==0
 keys=['signal','si_code','sender_pid_in_receiver_namespace','sender_uid','receiver_pid','receiver_tid','monotonic_ns','caller_address','installed_restorer_address','current_handler_address','current_flags','siginfo_verified']
 rows=[dict(zip(keys,FORMAT.unpack_from(data,i)))for i in range(0,len(data),FORMAT.size)]
 for row in rows:
  row['siginfo_verified']=bool(row['siginfo_verified'])
  row['sender_verified']=row['siginfo_verified']and row['si_code']in(0,-1,-6)
  if not row['siginfo_verified']:row['si_code']=None
  if not row['sender_verified']:
   row['sender_uid']=None;row['sender_pid_in_receiver_namespace']=None
 return rows
