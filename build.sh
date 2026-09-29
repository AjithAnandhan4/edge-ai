#!/bin/bash
# Assuming you have activated your zephyr virtual environment:
# source ~/zephyrproject/.venv/bin/activate

west build -b frdm_mcxn236 -p always .
echo "Build finished. To flash the board, run:"
echo "west flash"
