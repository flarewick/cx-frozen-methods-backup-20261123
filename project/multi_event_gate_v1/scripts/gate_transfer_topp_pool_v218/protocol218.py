"""Private local IPC, exact original TOPP arrays; no stochastic or GPU work."""
import socket,pickle,struct
MAX_BYTES=64*1024*1024
def receive(sock):
 def read(n):
  data=bytearray()
  while len(data)<n:
   piece=sock.recv(n-len(data))
   if not piece:raise EOFError('CPU TOPP worker closed')
   data.extend(piece)
  return data
 n=struct.unpack('!I',read(4))[0]
 if n>MAX_BYTES:raise ValueError('CPU TOPP payload exceeds limit')
 return pickle.loads(read(n))
def send(sock,value):
 payload=pickle.dumps(value,protocol=5)
 if len(payload)>MAX_BYTES:raise ValueError('CPU TOPP payload exceeds limit')
 sock.sendall(struct.pack('!I',len(payload))+payload)
