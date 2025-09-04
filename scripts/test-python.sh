#!/bin/bash
# TensorFuse Python Test Script
# This script runs only Python tests

exec "$(dirname "$0")/test.sh" --python-only "$@" 