#!/bin/bash
# Quick start script for memristor neural network experiments

echo "=== Memristor Neural Network Framework - Quick Start ==="
echo ""

# Set default values
CONFIG_DIR="configs"
OUTPUT_DIR="outputs"

# Function to run baseline experiment
run_baseline() {
    echo "Running baseline ResNet-20 experiment..."
    python -m src train --config ${CONFIG_DIR}/resnet20_baseline.yaml
    echo "Baseline experiment completed!"
    echo ""
}

# Function to run memristor no compensation experiment
run_memristor_no_comp() {
    echo "Running memristor ResNet-20 experiment (no compensation)..."
    python -m src train --config ${CONFIG_DIR}/resnet20_memristor_no_comp.yaml
    echo "Memristor no-comp experiment completed!"
    echo ""
}

# Function to run memristor with compensation experiment
run_memristor_comp() {
    echo "Running memristor ResNet-20 experiment (with HAT compensation)..."
    python -m src train --config ${CONFIG_DIR}/resnet20_memristor_comp.yaml --compensation hat
    echo "Memristor compensation experiment completed!"
    echo ""
}

# Function to evaluate a checkpoint
evaluate_checkpoint() {
    local checkpoint_path=$1
    if [ -z "$checkpoint_path" ]; then
        echo "Usage: evaluate_checkpoint <checkpoint_path>"
        return 1
    fi
    
    echo "Evaluating checkpoint: $checkpoint_path"
    python -m src eval --config ${CONFIG_DIR}/resnet20_baseline.yaml --checkpoint "$checkpoint_path"
    echo "Evaluation completed!"
    echo ""
}

# Main menu
echo "Select an option:"
echo "1) Run baseline experiment"
echo "2) Run memristor experiment (no compensation)"
echo "3) Run memristor experiment (with compensation)"
echo "4) Run all experiments"
echo "5) Evaluate checkpoint"
echo "6) Exit"
echo ""

read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        run_baseline
        ;;
    2)
        run_memristor_no_comp
        ;;
    3)
        run_memristor_comp
        ;;
    4)
        run_baseline
        run_memristor_no_comp
        run_memristor_comp
        echo "All experiments completed!"
        ;;
    5)
        read -p "Enter checkpoint path: " checkpoint
        evaluate_checkpoint "$checkpoint"
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "Done!"


