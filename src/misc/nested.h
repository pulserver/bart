
#if defined(__clang__) && !defined(__CUDACC__) && !defined(__HIPCC__)
#define NESTED(RET, NAME, ARGS) \
	RET (^NAME)ARGS = ^ARGS
#define CLOSURE_TYPE(RET, ARGS) typeof(RET (^) ARGS)
#define CLOSURE(T, x) x
#define CLOSURE_NULL_P(x) (NULL == (x))
#else
#if __GNUC__ >= 17
#define CLOSURE_TYPE(RET, ARGS) 		\
struct {					\
	typeof(RET (*) ARGS) code;		\
	void *chain;				\
}
#define NESTED(RET, NAME, ARGS) 		\
	RET NAME ARGS
#define CLOSURE(T, x) (T){ __builtin_call_code_address(x), __builtin_call_static_chain(x) }
#define CLOSURE_NULL_P(x) (NULL == (x).code)
#else
#define NESTED(RET, NAME, ARGS) \
	RET NAME ARGS
#define CLOSURE_TYPE(RET, ARGS) __typeof(RET (*) ARGS)
#define CLOSURE(T, x) x
#define CLOSURE_NULL_P(x) (NULL == (x))
#endif
#define __block
#endif

#if defined(__clang__) || !defined(NOEXEC_STACK)
#if __GNUC__ >= 17
#define NESTED_CALL(x, args)	__builtin_call_with_static_chain((x).code args, (x).chain)
#else
#define NESTED_CALL(x, args)	((x)args)
#endif
#else
#ifndef __x86_64__
#error NOEXEC_STACK only supported on x86_64
#endif
#include <stdio.h>
#if __GNUC__ >= 5
#define NESTED_CALL(p, args) ({												\
		__auto_type __p = (p);											\
		struct { unsigned short mov1; unsigned int addr; unsigned short mov2; void* chain; unsigned int jmp; } 	\
			__attribute__((packed))* __t = (void*)p;							\
		assert((0xbb41 == __t->mov1) && (0xba49 == __t->mov2) && (0x90e3ff49 == __t->jmp));			\
		__builtin_call_with_static_chain(((__typeof__(__p))((unsigned long)__t->addr))args, (void*)__t->chain);	\
	})
#else
#define NESTED_CALL(p, args) ({												\
		__auto_type __p = (p);											\
		struct { unsigned short mov1; void* addr; unsigned short mov2; void* chain; unsigned int jmp; } 	\
			__attribute__((packed))* __t = (void*)p;							\
		assert((0xbb49 == __t->mov1) && (0xba49 == __t->mov2) && (0x90e3ff49 == __t->jmp));			\
		__builtin_call_with_static_chain(((__typeof__(__p))(__t->addr))args, __t->chain);			\
	})
#endif
#endif

