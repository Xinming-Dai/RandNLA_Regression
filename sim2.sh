#!/bin/bash
# Define parameter values
n=1000
dist_values=("laplace" "point_mass" "cauchy" "gamma" "normal")
n_trials=100
delta_values=(0.01 0.05 0.1 0.2)
sketch_values=("OLS")
tau=(1.35 0)
k_values=(20 40 80 160)

# Iterate through all combinations
for k in "${k_values[@]}"; do
    for dist in "${dist_values[@]}"; do
        for delta in "${delta_values[@]}"; do
            for sketch in "${sketch_values[@]}"; do
            echo "Running simulation for n=$n, k=$k, dist=$dist, delta=$delta, sketch=$sketch"
            python simulation.py --n "$n" --p "20" --dist "$dist" --param "1" --n_trials "$n_trials" --delta "$delta" --sketch "$sketch" --k "$k"
            done
        done
    done
done

echo "All simulations completed!"