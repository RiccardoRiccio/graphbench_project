# Extending GraphBench Baselines for Electronic Circuit Performance Prediction

Code for the mini-project of L65 Geometric Deep Learning.

## Summary
This project extends GraphBench on the Electronic Circuits domain, benchmarking several graph models for predicting:
- power conversion efficiency
- voltage conversion ratio

The project includes both GraphBench-style residual implementations and original model implementations, and also evaluates cross-size generalization across 5-, 7-, and 10-component circuits.

## Models
- GCN
- GAT
- GIN
- GT
- GPS
- PNA
- MLP

## Repository structure
- `models/`: model implementations
- `scripts/`: training scripts

## Notes
This repository contains the core code used for the project report. Experimental outputs, intermediate results, and non-essential files were kept out of `main` for clarity.
