@echo off
cd /d "%~dp0"
uv run python -m solar_seed.cli %*
