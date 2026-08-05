function plot_sweep_mae_ratio(sweep, file_name, riceSweep, gorillaSweep)
    % plot_sweep_mae_ratio(sweep, file_name, riceSweep, gorillaSweep)
    %
    % riceSweep/gorillaSweep (opcionales, [] por defecto): tablas ya cargadas
    % con readtable() a partir de sweep_rice_results.csv / sweep_gorilla_results.csv
    % (una fila por ventana, con columnas 'compressed_bits' y 'mae'). Puedes
    % pasar solo una, las dos, o ninguna -- cada una se superpone de forma
    % independiente si se proporciona, situada en un "latent_dim equivalente"
    % = mean(compressed_bits)/32 (misma unidad -- floats de 32 bits -- que
    % usa el latent_dim del AE), para poder compararla en el mismo eje X.
    if nargin < 3
        riceSweep = [];
    end
    if nargin < 4
        gorillaSweep = [];
    end
    show_rice = ~isempty(riceSweep);
    show_gorilla = ~isempty(gorillaSweep);

    % Size parameters for plot
    col_width = 3.5;
    fig_height = 2.35;
    fig = figure('Units','inches','Position',[1 1 col_width fig_height]);
    set(fig,'PaperUnits','inches');
    set(fig,'PaperSize',[col_width fig_height]);
    set(fig,'PaperPosition',[0 0 col_width fig_height]);
    set(fig,'PaperPositionMode','manual');

    %Font size
    custom_fontsize = 8;
    COLOR_SYM     = [0.00, 0.45, 0.74];   % azul    -> simetrico
    COLOR_ASYM    = [0.85, 0.33, 0.10];   % naranja -> asimetrico
    COLOR_RICE    = [0.49, 0.18, 0.56];   % morado  -> Rice-Golomb
    COLOR_GORILLA = [0.30, 0.60, 0.30];   % verde   -> Gorilla
    BITS_PER_VALUE = 32;                  % float32, igual criterio que en main.py

    latentDims = sort(unique(sweep.latent_dim));

    % Tiled layout ratio left, MAE right
    t = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');
    axRatio = nexttile; hold(axRatio, 'on');
    axMAE = nexttile; hold(axMAE, 'on');

    % Global title
    figTitle = 'Compression ratio and MAE vs latent dimension';
    sgtitle(t, figTitle, 'FontSize', custom_fontsize, 'FontWeight', 'normal');

    % Manage global legend
    n_entries = 2 + show_rice + show_gorilla;
    legend_handles = gobjects(1, n_entries);
    legend_entries = cell(1, n_entries);
    legend_entries(1:2) = {'Symmetric', 'Asymmetric'};
    labels = [true, false];
    colors = {COLOR_SYM, COLOR_ASYM};

    % For loop: curvas del AE (dependen de latent_dim)
    for i = 1:numel(labels)
        symVal = labels(i);

        ratio_mean_v = zeros(numel(latentDims), 1); ratio_ci_v = zeros(numel(latentDims), 1);
        mae_mean_v   = zeros(numel(latentDims), 1); mae_ci_v   = zeros(numel(latentDims), 1);

        for k = 1:numel(latentDims)
            mask = sweep.symmetric == symVal & sweep.latent_dim == latentDims(k);

            ratio_vals = sweep.ratio(mask);
            mae_vals   = sweep.mae_mean(mask);

            [ratio_mean_v(k), ratio_ci_v(k)] = ConfidenceInterval(ratio_vals);
            [mae_mean_v(k), mae_ci_v(k)]     = ConfidenceInterval(mae_vals);
        end
        if symVal
            lineSpec = '-o';
        else
            lineSpec = '--s';
        end

        h = errorbar(axRatio, latentDims, ratio_mean_v, ratio_ci_v, lineSpec, 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axMAE, latentDims, mae_mean_v, mae_ci_v, lineSpec, 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i) = h;
    end

    % Baselines clasicos (Rice-Golomb, Gorilla): no tienen un latent_dim de
    % diseno, pero podemos calcular un "latent_dim equivalente" a partir de
    % los bits que ocupan de verdad: mean(compressed_bits)/32. Se promedia
    % compressed_bits directamente (no el ratio) porque ratio=1/compressed_bits
    % (salvo constantes) y promediar un ratio y luego invertirlo NO da lo
    % mismo que promediar los bits y convertir despues (desigualdad de
    % Jensen) -- promediar los bits es la cantidad fisica real (cuantos bits
    % hace falta transmitir de media), que es lo que queremos representar.
    %
    % OJO: en el panel de ratio esto sigue siendo tautologico -- el ratio que
    % se dibuja ahi (input_dim/latent_eq) se deriva del mismo latent_eq que
    % define la posicion en X, asi que el punto cae por construccion sobre la
    % curva ratio=input_dim/latent_dim del AE. Se deja solo como referencia
    % visual de donde cae ese tamano de codigo en el eje compartido. La
    % comparacion real, con contenido empirico, es la del panel de MAE.
    next_idx = 3;
    input_dim = sweep.input_dim(1);

    if show_rice
        [rice_mae_mean, rice_mae_ci] = ConfidenceInterval(riceSweep.mae);
        [rice_bits_mean, ~]          = ConfidenceInterval(riceSweep.compressed_bits);
        rice_latent_eq               = rice_bits_mean / BITS_PER_VALUE;
        rice_ratio_eq                = input_dim / rice_latent_eq;

        plot(axRatio, rice_latent_eq, rice_ratio_eq, 'pentagram', 'Color', COLOR_RICE, 'MarkerFaceColor', COLOR_RICE, 'MarkerSize', 6, 'LineWidth', 1.2);
        legend_handles(next_idx) = errorbar(axMAE, rice_latent_eq, rice_mae_mean, rice_mae_ci, 'pentagram', 'Color', COLOR_RICE, 'MarkerFaceColor', COLOR_RICE, 'MarkerSize', 6, 'LineWidth', 1.2);
        legend_entries{next_idx} = 'Rice-Golomb';
        next_idx = next_idx + 1;
    end

    if show_gorilla
        [gorilla_mae_mean, gorilla_mae_ci] = ConfidenceInterval(gorillaSweep.mae);
        [gorilla_bits_mean, ~]             = ConfidenceInterval(gorillaSweep.compressed_bits);
        gorilla_latent_eq                  = gorilla_bits_mean / BITS_PER_VALUE;
        gorilla_ratio_eq                   = input_dim / gorilla_latent_eq;

        plot(axRatio, gorilla_latent_eq, gorilla_ratio_eq, 'hexagram', 'Color', COLOR_GORILLA, 'MarkerFaceColor', COLOR_GORILLA, 'MarkerSize', 6, 'LineWidth', 1.2);
        legend_handles(next_idx) = errorbar(axMAE, gorilla_latent_eq, gorilla_mae_mean, gorilla_mae_ci, 'hexagram', 'Color', COLOR_GORILLA, 'MarkerFaceColor', COLOR_GORILLA, 'MarkerSize', 6, 'LineWidth', 1.2);
        legend_entries{next_idx} = 'Gorilla';
        next_idx = next_idx + 1;
    end

    hold(axRatio, 'off'); hold(axMAE, 'off');
    xlabel(axRatio, 'Latent dimension'); ylabel(axRatio, 'Compression ratio');
    xlabel(axMAE, 'Latent dimension'); ylabel(axMAE, 'MAE (°C)');
    set(axRatio, 'FontSize', custom_fontsize);
    set(axMAE, 'FontSize', custom_fontsize);

    % Leyenda global fuera, abajo
    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal', 'FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', min(n_entries, 2));
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    % Export file to PDF
    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
