

// clock_gettime not defined on osx
#if defined(__MACH__) && !defined(CLOCK_MONOTONIC)
#include <sys/time.h>
#define CLOCK_MONOTONIC 0
// clock_gettime is not implemented on older versions of OS X (< 10.12).
// If implemented, CLOCK_REALTIME will have already been defined.
int clock_gettime(int clk_id, struct timespec *t)
{
  struct timeval now;
  int rv = gettimeofday(&now, NULL);
  if (rv)
    return rv;
  t->tv_sec = now.tv_sec;
  t->tv_nsec = now.tv_usec * 1000;
  return 0;
}
#endif

  /*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>
#include <math.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif

stack_tt *stack;
node_tt *nodes[NB_THREADS];

data_t data;

struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

#if MEASURE != 0

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < (MAX_PUSH_POP / NB_THREADS); i++)
      {
        stack_pop(stack);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
    
  }


#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
      stack_push(stack, i);
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_tt));
  stack_init(stack);

}

void
test_teardown()
{
  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
  
}

// Push from one thread
void *thread_stack_push(void* arg){
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int id = args->id;

  size_t i;
  for (i = 0; i < (MAX_PUSH_POP/NB_THREADS); i++)
  {
    stack_push(stack, id);
  }

  return NULL;
}

int
test_push_safe()
{
  stack_measure_arg_t args[NB_THREADS];
  // Initialize threads
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  size_t i;
  for (i = 0; i < NB_THREADS; i++)
  {
    args[i].id = i;
    pthread_create(&thread[i], &attr, &thread_stack_push, (void *)&args[i]);
  }

  // Wait for all threads to finish before proceeding
  for(i=0; i<NB_THREADS; i++){
      pthread_join(thread[i], NULL);
  }

  int stack_sum = 0;

  node_tt *tmp = stack->head;
  while(tmp != NULL){
    stack_sum += tmp->value;
    tmp = tmp->next;
  }
  free(tmp);

  int real_sum = 0;
  for(int i=0; i<NB_THREADS; i++){
    real_sum += i * (MAX_PUSH_POP / NB_THREADS);
  }


  // check if the stack is in a consistent state
  int res = assert(stack_sum == real_sum);
  return res;
}




// Pop from one thread
void *thread_stack_pop(void *arg)
{
  // stack_measure_arg_t *args = (stack_measure_arg_t *)arg;
  // int id = args->id;

  size_t i;
  for (i = 0; i < (MAX_PUSH_POP / NB_THREADS) + 1; i++)
  {
    stack_pop(stack);
  }

  return NULL;
}

int
test_pop_safe()
{

  // Fill stack with values;
  size_t i;
  for(int i=0; i<MAX_PUSH_POP; i++){
    stack_push(stack, i);
  }

  // Ensure that the stack has been initialized correctly
  node_tt *ptr = malloc(sizeof(node_tt));
  ptr = stack->head;
  int count = 0;
  while(ptr){
    count++;
    ptr = ptr->next;
  }

  if(count != MAX_PUSH_POP)
    return 0;

  // Start poping nodes
  stack_measure_arg_t args[NB_THREADS];
  // Initialize threads
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  for (i = 0; i < NB_THREADS; i++)
  {
    args[i].id = i;
    pthread_create(&thread[i], &attr, &thread_stack_pop, (void *)&args[i]);
  }

  // Wait for all threads to finish before proceeding
  for(i=0; i<NB_THREADS; i++){
      pthread_join(thread[i], NULL);
  }


  int res = assert(stack->head == NULL);
  return res;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success, aba_detected = 0;
  // Write here a test for the ABA problem
  success = aba_detected;
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);

  test_setup();
  


  #if MEASURE == 1
    
  for (i = 0; i < MAX_PUSH_POP; i++) {
      stack_push(stack, i);
    }
    
  #endif
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }


  for (i = 0; i < NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }
  
  clock_gettime(CLOCK_MONOTONIC, &stop);
  

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}
