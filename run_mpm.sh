#!/bin/bash
# Simple script to run MPM simulations

echo "MPM Simulation Framework"
echo "======================="

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 config/multimaterial/multimaterial_1_1.yaml"
    exit 1
fi

CONFIG_FILE=$1

echo "Running MPM simulation with config: $CONFIG_FILE"
python GNN_particles_Ntype.py --config $CONFIG_FILE
