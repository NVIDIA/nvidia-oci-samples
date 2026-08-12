# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.PHONY: help modes test

help:
	@echo "make modes  List supported benchmark modes and containers"
	@echo "make test   Run repository unit tests"

modes:
	python3 benchmark.py --list-modes

test:
	python3 -m compileall -q benchmark.py benchmarks tools tests
	python3 -m unittest discover -s tests -v
