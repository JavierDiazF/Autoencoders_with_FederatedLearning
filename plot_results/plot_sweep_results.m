%% plot_sweep_results.m
% Lee results/sweep_a_results.csv y results/sweep_b_results.csv (generados
% por main.py) y genera las graficas de comparacion:
%
%   Figura 1: Sweep A -- MSE vs ratio de compresion (una subgrafica por input_dim)
%   Figura 2: Sweep A -- MAE vs ratio de compresion (una subgrafica por input_dim)
%   Figura 3: Sweep B -- energia estimada (encoder y decoder) vs num. de hidden layers
%   Figura 4: Sweep B -- MSE vs num. de hidden layers
%
% En todas las graficas se comparan modelos simetricos vs. asimetricos,
% agregando (media +/- desviacion tipica) sobre las 10 semillas entrenadas
% por configuracion.

clear; clc; close all;

RESULTS_DIR = '../results/azure_results';
ENERGY_PJ_PER_MAC = 4.6;   % Horowitz, ISSCC 2014, 45nm float32 mult+add (pJ/MAC)

COLOR_SYM  = [0.00, 0.45, 0.74];   % azul   -> simetrico
COLOR_ASYM = [0.85, 0.33, 0.10];   % naranja -> asimetrico

%% ------------------------------------------------------------------
%  Carga de datos
%% ------------------------------------------------------------------
sweepA = readtable(fullfile(RESULTS_DIR, 'sweep_a_results.csv'));
sweepB = readtable(fullfile(RESULTS_DIR, 'sweep_b_results.csv'));
sweepC = readtable(fullfile(RESULTS_DIR, 'sweep_a_results.csv'));

sweepA.symmetric = parseBoolColumn(sweepA.symmetric);
sweepB.symmetric = parseBoolColumn(sweepB.symmetric);
sweepC.symmetric = parseBoolColumn(sweepC.symmetric);

%% ------------------------------------------------------------------
%  Figura 1 y 2: Sweep A -- MSE y MAE vs ratio de compresion,
%  una subgrafica por input_dim
%% ------------------------------------------------------------------
inputDims = unique(sweepA.input_dim);

plotSweepAMetric(sweepA, inputDims, 'mse_mean', 'MSE (°C^2)', ...
    'Sweep A: MSE vs ratio de compresion', COLOR_SYM, COLOR_ASYM);

plotSweepAMetric(sweepA, inputDims, 'mae_mean', 'MAE (°C)', ...
    'Sweep A: MAE vs ratio de compresion', COLOR_SYM, COLOR_ASYM);
%% ------------------------------------------------------------------
%  Figura 1 y 2: Sweep C -- MSE y MAE vs ratio de compresion,
%  una subgrafica por input_dim
%% ------------------------------------------------------------------
inputDims = unique(sweepC.input_dim);

plotSweepAMetric(sweepC, inputDims, 'mse_mean', 'MSE (°C^2)', ...
    'Sweep C: MSE vs ratio de compresion', COLOR_SYM, COLOR_ASYM);

plotSweepAMetric(sweepC, inputDims, 'mae_mean', 'MAE (°C)', ...
    'Sweep C: MAE vs ratio de compresion', COLOR_SYM, COLOR_ASYM);

%% ------------------------------------------------------------------
%  Figura 3: Sweep B -- energia (encoder y decoder por separado) vs
%  numero de hidden layers
%% ------------------------------------------------------------------
sweepB.encoder_energy_nJ = sweepB.encoder_macs * ENERGY_PJ_PER_MAC / 1000;
sweepB.decoder_energy_nJ = sweepB.decoder_macs * ENERGY_PJ_PER_MAC / 1000;

figure('Name', 'Sweep B - Energia vs hidden layers', 'Color', 'w');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plotBySymmetric(sweepB, 'hidden_layers', 'encoder_energy_nJ', COLOR_SYM, COLOR_ASYM);
xlabel('Numero de hidden layers');
ylabel('Energia estimada del encoder (nJ)');
title('Encoder (lado del sensor)');
legend('Simetrico', 'Asimetrico', 'Location', 'best');
grid on;

nexttile;
plotBySymmetric(sweepB, 'hidden_layers', 'decoder_energy_nJ', COLOR_SYM, COLOR_ASYM);
xlabel('Numero de hidden layers');
ylabel('Energia estimada del decoder (nJ)');
title('Decoder (lado del receptor)');
legend('Simetrico', 'Asimetrico', 'Location', 'best');
grid on;

sgtitle('Sweep B: energia estimada (MACs x 4.6 pJ/MAC) vs profundidad');

%% ------------------------------------------------------------------
%  Figura 4: Sweep B -- MSE vs numero de hidden layers
%% ------------------------------------------------------------------
figure('Name', 'Sweep B - MSE vs hidden layers', 'Color', 'w');
plotBySymmetric(sweepB, 'hidden_layers', 'mse_mean', COLOR_SYM, COLOR_ASYM);
xlabel('Numero de hidden layers');
ylabel('MSE (°C^2)');
title('Sweep B: MSE vs profundidad');
legend('Simetrico', 'Asimetrico', 'Location', 'best');
grid on;


%% ==================================================================
%  Funciones locales
%% ==================================================================

function boolCol = parseBoolColumn(col)
    % pandas.to_csv escribe los booleanos de Python como texto "True"/"False",
    % no como 0/1 ni "true"/"false" en minuscula -- readtable puede
    % importar esa columna como cell de char, string o categorical segun
    % la version, nunca como logical automaticamente. Se normaliza aqui.
    if islogical(col)
        boolCol = col;
    else
        boolCol = string(col) == "True";
    end
end

function plotSweepAMetric(sweepA, inputDims, valueVar, yLabelStr, figTitle, colorSym, colorAsym)
    figure('Name', figTitle, 'Color', 'w');
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    for k = 1:numel(inputDims)
        nexttile;
        thisDim = sweepA(sweepA.input_dim == inputDims(k), :);

        summaryT = groupsummary(thisDim, {'latent_dim', 'symmetric'}, {'mean', 'std'}, valueVar);
        summaryT.ratio = inputDims(k) ./ (summaryT.latent_dim + 3);

        plotGroupedErrorbar(summaryT, 'ratio', valueVar, colorSym, colorAsym);

        xlabel('Ratio de compresion (input\_dim / (latent\_dim+3))');
        ylabel(yLabelStr);
        title(sprintf('input\\_dim = %d', inputDims(k)));
        legend('Simetrico', 'Asimetrico', 'Location', 'best');
        grid on;
    end
    sgtitle(figTitle);
end

function plotBySymmetric(T, xVar, valueVar, colorSym, colorAsym)
    summaryT = groupsummary(T, {xVar, 'symmetric'}, {'mean', 'std'}, valueVar);
    plotGroupedErrorbar(summaryT, xVar, valueVar, colorSym, colorAsym);
end

function plotGroupedErrorbar(summaryT, xVar, valueVar, colorSym, colorAsym)
    % summaryT viene de groupsummary: contiene columnas
    % mean_<valueVar> y std_<valueVar> ademas de 'symmetric' y xVar.
    meanCol = ['mean_' valueVar];
    stdCol  = ['std_' valueVar];

    hold on;
    labels = [true, false];
    colors = {colorSym, colorAsym};
    for i = 1:numel(labels)
        subset = summaryT(summaryT.symmetric == labels(i), :);
        subset = sortrows(subset, xVar);
        errorbar(subset.(xVar), subset.(meanCol), subset.(stdCol), '-o', ...
            'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.5);
    end
    hold off;
end
