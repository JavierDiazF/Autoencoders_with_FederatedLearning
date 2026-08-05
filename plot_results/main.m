%% Script to plot A results 
%
%   [+] Original Author:    Javier Díaz <j.diazf@edu.uah.es> 
%
%   [+] Date: 30 March 2026

clc
close all
clear variables

%% Global vars
%RESULTS_DIR = 'results/azure_results/n30iterations';
RESULTS_DIR = 'results';
FIGURES_DIR = '../Figures';
MATLAB_RESULTS_DIR = 'matlab_results';
SWEEP_A_RESULTS = 'sweep_a_results.mat';
SWEEP_B_RESULTS = 'sweep_b_results.mat';
SWEEP_C_RESULTS = 'sweep_c_results.mat';
SWEEP_D_RESULTS = 'sweep_d_results.mat';
ENERGY_PJ_PER_MAC = 4.6;   % Horowitz, ISSCC 2014, 45nm float32 mult+add (pJ/MAC)
HIDDEN_LAYERS_MODE = {'encoder_decoder', 'mse_encoder', 'mae_encoder'};

COLOR_SYM  = [0.00, 0.45, 0.74];   % azul   -> simetrico
COLOR_ASYM = [0.85, 0.33, 0.10];   % naranja -> asimetrico

%% Load and save data
%{
if (isfile(fullfile(MATLAB_RESULTS_DIR, SWEEP_A_RESULTS)) && isfile(fullfile(MATLAB_RESULTS_DIR, SWEEP_B_RESULTS)) && isfile(fullfile(MATLAB_RESULTS_DIR, SWEEP_C_RESULTS)))
    % File exist.
    disp("Getting data from stored variables.")
    load(fullfile(MATLAB_RESULTS_DIR, SWEEP_A_RESULTS));
    load(fullfile(MATLAB_RESULTS_DIR, SWEEP_B_RESULTS));
    load(fullfile(MATLAB_RESULTS_DIR, SWEEP_C_RESULTS));
else
    sweepA = readtable(fullfile('..', RESULTS_DIR, 'sweep_a_results.csv'));
    sweepB = readtable(fullfile('..', RESULTS_DIR, 'sweep_b_results.csv'));
    sweepC = readtable(fullfile('..', RESULTS_DIR, 'sweep_c_results.csv'));
    save(fullfile(MATLAB_RESULTS_DIR, SWEEP_A_RESULTS), "sweepA")
    save(fullfile(MATLAB_RESULTS_DIR, SWEEP_B_RESULTS), "sweepB")
    save(fullfile(MATLAB_RESULTS_DIR, SWEEP_C_RESULTS), "sweepC")
end
%}
%sweepA = readtable(fullfile('..', RESULTS_DIR, 'sweep_a_results.csv'));
sweepB = readtable(fullfile('..', RESULTS_DIR, 'sweep_b_results.csv'));
sweepC = readtable(fullfile('..', RESULTS_DIR, 'sweep_c_results.csv'));
sweepD = readtable(fullfile('..', RESULTS_DIR, 'sweep_d_results.csv'));
sweepRice = readtable(fullfile('..', RESULTS_DIR, 'sweep_rice_results.csv'));
sweepGorilla = readtable(fullfile('..', RESULTS_DIR, 'sweep_gorilla_results.csv'));
%save(fullfile(MATLAB_RESULTS_DIR, SWEEP_A_RESULTS), "sweepA")
%save(fullfile(MATLAB_RESULTS_DIR, SWEEP_B_RESULTS), "sweepB")
%save(fullfile(MATLAB_RESULTS_DIR, SWEEP_C_RESULTS), "sweepC")
%save(fullfile(MATLAB_RESULTS_DIR, SWEEP_D_RESULTS), "sweepD")
%% Post processing data
% Parse "True/False" columns into boolean
%sweepA.symmetric = parseBoolColumn(sweepA.symmetric);
sweepB.symmetric = parseBoolColumn(sweepB.symmetric);
sweepC.symmetric = parseBoolColumn(sweepC.symmetric);
sweepD.symmetric = parseBoolColumn(sweepD.symmetric);

%% Plot SweepC
%plot_sweep_mae_mse(sweepC, fullfile(FIGURES_DIR, 'sweepc_plotted'))
plot_sweep_mae(sweepC, fullfile(FIGURES_DIR, 'sweepc_mae_plotted_javi'))

%% Plot SweepB
% Plot decoder and encoder energy
%plot_sweep_hidden_layers(sweepB, fullfile(FIGURES_DIR, 'encoder_decoder'), HIDDEN_LAYERS_MODE{1})
% Pot encoder energy against MSE
%plot_sweep_hidden_layers(sweepB, fullfile(FIGURES_DIR, 'encoder_energy_mse'), HIDDEN_LAYERS_MODE{2})
%plot_sweep_hidden_layers(sweepB, fullfile(FIGURES_DIR, 'encoder_energy_mae'), HIDDEN_LAYERS_MODE{3})
plot_sweep_hidden_layers_triangle(sweepB, fullfile(FIGURES_DIR, 'encoder_decoder_mae_javi'))
%% Plot SweepD
%plot_sweep_mae_mse(sweepC, fullfile(FIGURES_DIR, 'sweepc_plotted'))
%plot_sweep_mae_ratio(sweepD, fullfile(FIGURES_DIR, 'sweepd_mae_ratio_plotted'))
plot_sweep_mae_ratio_2x2(sweepD, fullfile(FIGURES_DIR, 'sweep_mae_ratio_2x2_javi'), sweepRice, sweepGorilla);

%% Plot with Rice-Golomb andd Gorilla commpression techniques
%riceSweep = readtable(fullfile('..', RESULTS_DIR,'sweep_rice_results.csv'));
%gorillaSweep = readtable(fullfile('..', RESULTS_DIR, 'sweep_gorilla_results.csv'));
%plot_sweep_mae_ratio(sweepD, fullfile(FIGURES_DIR, 'sweepd_vs_rice'), riceSweep);
%plot_sweep_mae_ratio(sweepD, fullfile(FIGURES_DIR, 'sweepd_vs_gorilla'), [], gorillaSweep);
%plot_sweep_mae_ratio(sweepD, fullfile(FIGURES_DIR, 'sweepd_vs_baselines'), riceSweep, gorillaSweep);

