#!/bin/sh

valgrind --error-exitcode 1 -s $1
exit $?
