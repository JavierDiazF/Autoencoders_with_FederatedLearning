%% Script to plot IIoT AE-vs-raw compression experiment results
% (Autoencoders/iiot_compression_experiment.py) -- separate from main.m,
% which plots the unrelated lake-temperature compression benchmark.
%
%   [+] Date: 5 August 2026

clc
close all
clear variables

RESULTS_DIR = 'results/initial_k15';
FIGURES_DIR = '../Figures';

latentSweep = readtable(fullfile('..', RESULTS_DIR, 'iiot_latent_sweep.csv'));
latentSweep.symmetric = parseBoolColumn(latentSweep.symmetric);

plot_iiot_latent_sweep(latentSweep, fullfile(FIGURES_DIR, 'iiot_latent_sweep_plotted'));

%% Ratio sweep -- F1 vs latent_dim, colored by target ratio (full
% payload-vs-ratio view still on hold, see conversation)
ratioSweep = readtable(fullfile('..', RESULTS_DIR, 'iiot_ratio_sweep.csv'));
ratioSweep.symmetric = parseBoolColumn(ratioSweep.symmetric);

plot_iiot_ratio_sweep_f1(ratioSweep, fullfile(FIGURES_DIR, 'iiot_ratio_sweep_f1_plotted'));
