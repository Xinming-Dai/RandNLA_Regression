#!/bin/bash
# Define parameter values
n=1000
dist_values=("laplace" "point_mass" "cauchy" "gamma" "normal")
n_trials=100
delta_values=(0.01 0.05 0.1 0.2)
sketch_values=("sparse_sign" "clarkson_woodruff" "uniform_sparse" "normal" "uniform_dense" "proposal1" "proposal2")
tau=(1.35 0)
k_values=(20 40 80 160)

# Iterate through all combinations
for k in "${k_values[@]}"; do
    for dist in "${dist_values[@]}"; do
        for delta in "${delta_values[@]}"; do
            for sketch in "${sketch_values[@]}"; do
            echo "Running simulation for n=$n, k=$k, dist=$dist, delta=$delta, sketch=$sketch"
                if [ "$sketch" == "proposal1" ]; then
                    for t in "${tau[@]}"; do
                        python simulation.py --n "$n" --p "20" --dist "$dist" --param "1" --n_trials "$n_trials" --delta "$delta" --sketch "$sketch" --tau "$t" --k "$k"
                    done
                fi
                if [ "$sketch" == "proposal2" ]; then
                    for t in "${tau[@]}"; do
                        python simulation.py --n "$n" --p "20" --dist "$dist" --param "1" --n_trials "$n_trials" --delta "$delta" --sketch "$sketch" --tau "$t" --k "$k"
                    done
                fi
            done
        done
    done
done

echo "All simulations completed!"