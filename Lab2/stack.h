/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H


struct node {
  
  int value;
  struct node *next;
  
};
typedef struct node node_tt;

// Single linked list where tail = NULL
struct stack
{
  node_tt *head;

  #if NON_BLOCKING == 0
    // Lock_based stack
    pthread_mutex_t lock;

  #elif NON_BLOCKING == 1
    // CAS-based stack

  #endif
  
};
typedef struct stack stack_tt;

int stack_push(stack_tt *stack, node_tt *node);
int stack_pop(stack_tt *stack);
void stack_print(stack_tt *stack);
void stack_init(stack_tt *stack);

/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_tt *stack);
#endif /* STACK_H */
