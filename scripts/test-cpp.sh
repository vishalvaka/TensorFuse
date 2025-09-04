#!/bin/bash
# TensorFuse C++ Test Script
# This script runs only C++ tests

exec "$(dirname "$0")/test.sh" --cpp-only "$@" 