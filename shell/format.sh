#!/bin/bash

uvx ruff format && uvx ruff check --fix --select I
