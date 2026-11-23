/* Linux x86_64, no libc. Record call provenance before reading siginfo. */
typedef long i64;
typedef unsigned long u64;
typedef int i32;
typedef unsigned int u32;
struct action { void *fn; u64 flags; void *restorer; u64 mask; };
struct stamp { i64 sec, ns; };
struct receipt {
 i32 signo, code, sender_pid; u32 sender_uid;
 i64 receiver_pid, receiver_tid, monotonic_ns;
 u64 caller, installed_restorer, current_handler, current_flags, siginfo_verified;
};
static struct action previous;
static int sink = -1;
static long call4(long n,long a,long b,long c,long d) {
 register long r10 __asm__("r10")=d;
 long result;
 __asm__ volatile("syscall":"=a"(result):"a"(n),"D"(a),"S"(b),"d"(c),"r"(r10):"rcx","r11","memory");
 return result;
}
static void record_and_chain(int signo,void *info,void *context) {
 struct stamp now;
 struct action current;
 struct receipt row;
 row.signo=signo;row.code=2147483647;row.sender_pid=-1;row.sender_uid=4294967295U;
 row.caller=(u64)__builtin_return_address(0);
 row.installed_restorer=(u64)previous.restorer;
 current.fn=0;current.flags=0;current.restorer=0;current.mask=0;
 call4(13,2,0,(long)&current,8);
 row.current_handler=(u64)current.fn;row.current_flags=current.flags;
 /* A library may retain this pointer and call it as handler(int).
    Such calls have no valid siginfo arguments: do not dereference them. */
 row.siginfo_verified=row.caller==row.installed_restorer &&
   current.fn==(void*)record_and_chain && (current.flags&4UL) && info && context;
 if(row.siginfo_verified) {
  if(*(i32*)info==signo) {
   row.code=*(i32*)((char*)info+8);
   if(row.code==0 || row.code==-1 || row.code==-6) {
    row.sender_pid=*(i32*)((char*)info+16);
    row.sender_uid=*(u32*)((char*)info+20);
   }
  } else row.siginfo_verified=0;
 }
 row.receiver_pid=call4(39,0,0,0,0);row.receiver_tid=call4(186,0,0,0,0);
 now.sec=0;now.ns=0;call4(228,1,(long)&now,0,0);
 row.monotonic_ns=now.sec*1000000000L+now.ns;
 if(sink>=0)call4(1,sink,(long)&row,sizeof(row),0);
 if(previous.flags&4UL)((void(*)(int,void*,void*))previous.fn)(signo,info,context);
 else ((void(*)(int))previous.fn)(signo);
}
long install_receipt(int fd) {
 struct action next;
 long result;
 if(sink>=0 || fd<0)return -1;
 result=call4(13,2,0,(long)&previous,8);if(result<0)return result;
 if(previous.fn==(void*)0 || previous.fn==(void*)1 || (previous.flags&4UL))return -2;
 next.fn=(void*)record_and_chain;next.flags=previous.flags|4UL;
 next.restorer=previous.restorer;next.mask=previous.mask;
 sink=fd;result=call4(13,2,(long)&next,0,8);
 if(result<0)sink=-1;
 return result;
}
long restore_receipt(void) {
 long result;
 if(sink<0)return -1;
 result=call4(13,2,(long)&previous,0,8);if(result==0)sink=-1;
 return result;
}
long receipt_record_size(void) { return sizeof(struct receipt); }
/* CPU fixture only: simulate a library's saved handler(int) invocation. */
void fixture_saved_one_argument_handler(void) {
 ((void(*)(int))record_and_chain)(2);
}
