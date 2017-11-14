/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

static node_tt nodePool[MAX_PUSH_POP];
_Atomic static size_t nodeNr;

void stack_init(stack_tt *stack)
{
  nodeNr = 0;
  stack->head = NULL;
  
  #if NON_BLOCKING == 0
    pthread_mutex_init(&stack->lock, NULL);
  #endif
}

void stack_print(stack_tt *stack){

  node_tt *tmp = stack->head;
  while(tmp != NULL){
    printf("%d\n", tmp->value);
    tmp = tmp->next;
  }
  free(tmp);


}

int
stack_check(stack_tt *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(stack_tt *stack, int value)
{
  node_tt* newNode = &nodePool[nodeNr++];
  newNode->value = value;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  int status = pthread_mutex_lock(&stack->lock);

  while(status != 0){
    status = pthread_mutex_lock(&stack->lock);
    printf("%d\n", status);
  }

  newNode->next = stack->head;
  stack->head = newNode;
  pthread_mutex_unlock(&stack->lock);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  
  node_tt* old;
  do {
    old = stack->head;
    newNode->next = old;
  } while (cas((size_t *)&stack->head, (size_t)old, (size_t)newNode) != (size_t)old);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_tt*)1);

  return 0;
}

int /* Return the type you prefer */
stack_pop(stack_tt *stack)
{
  if (stack->head == NULL){
    return -1;
  }

#if NON_BLOCKING == 0

  pthread_mutex_lock(&stack->lock);
  stack->head = stack->head->next;
  pthread_mutex_unlock(&stack->lock);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  node_tt* old;
  node_tt* next;

  do {
    old = stack->head;
    next = old->next;
  } while (cas((size_t*)&stack->head, (size_t)old, (size_t)next) != (size_t)old);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

