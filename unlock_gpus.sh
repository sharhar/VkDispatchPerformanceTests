#!/bin/bash
# 1. Reset Graphics Clock to default boost behavior
sudo nvidia-smi -rgc

# 2. Reset Memory Clock to default behavior
sudo nvidia-smi -rmc

# 3. Reset Power Limit (if modified)
sudo nvidia-smi -rpl

# 4. Disable Persistence Mode
sudo nvidia-smi -pm 0