function plot_sweep_mae_ratio_2x2(sweep, file_name, riceSweep, gorillaSweep)
    % Igual que plot_sweep_mae_ratio, pero en una rejilla 2x2:
    %   arriba-izda: ratio de compresion del AE vs latent_dim (sweep.latent_dim)
    %   arriba-dcha: MAE del AE vs latent_dim
    %   abajo-izda:  ratio de compresion de Rice-Golomb y Gorilla vs input_dim
    %   abajo-dcha:  MAE de Rice-Golomb y Gorilla vs input_dim

    col_width = 3.5;
    fig_height = 4;
    fig = figure('Units','inches','Position',[1 1 col_width fig_height]);
    set(fig,'PaperUnits','inches');
    set(fig,'PaperSize',[col_width fig_height]);
    set(fig,'PaperPosition',[0 0 col_width fig_height]);
    set(fig,'PaperPositionMode','manual');

    %Font size
    custom_fontsize = 8;
    %custom_fontsize = 7;
    COLOR_SYM     = [0.00, 0.45, 0.74];   % azul    -> simetrico
    COLOR_ASYM    = [0.85, 0.33, 0.10];   % naranja -> asimetrico
    COLOR_RICE    = [0.49, 0.18, 0.56];   % morado  -> Rice-Golomb
    COLOR_GORILLA = [0.30, 0.60, 0.30];   % verde   -> Gorilla

    % Tiled layout 2x2
    t = tiledlayout(2, 2, 'TileSpacing','compact', 'Padding','compact');
    % First two plots are the ratios
    axRatioAE  = nexttile; hold(axRatioAE, 'on');
    axRatioCls = nexttile; hold(axRatioCls, 'on');
    % Bottom plots are the MAE
    axMAE_AE   = nexttile; hold(axMAE_AE, 'on');
    axMAE_Cls  = nexttile; hold(axMAE_Cls, 'on');

    sgtitle(t, sprintf('AE vs. baseline compression: Compression ratio and MAE'), 'FontSize', custom_fontsize + 2, 'FontWeight', 'normal', 'Interpreter', 'tex');

    legend_handles = gobjects(1, 4);
    legend_entries = {'Symmetric', 'Asymmetric', 'Rice-Golomb', 'Gorilla'};

    % ---------------------------------------------------------------
    % Fila superior: curvas del AE, x = latent_dim
    % ---------------------------------------------------------------
    latentDims = sort(unique(sweep.latent_dim));
    labels = [true, false];
    ae_colors = {COLOR_SYM, COLOR_ASYM};

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

        h = errorbar(axRatioAE, latentDims, ratio_mean_v, ratio_ci_v, lineSpec, 'Color', ae_colors{i}, 'MarkerFaceColor', ae_colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axMAE_AE, latentDims, mae_mean_v, mae_ci_v, lineSpec, 'Color', ae_colors{i}, 'MarkerFaceColor', ae_colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i) = h;
    end
    % Comment axRatioAE xlabel because it can be shown with xlabel of
    % axMAE_AE
    %xlabel(axRatioAE, 'Latent dimension'); 
    ylabel(axRatioAE, 'Compression ratio', 'FontSize', custom_fontsize);
    xlabel(axMAE_AE, 'Latent dimension', 'FontSize', custom_fontsize);  
    ylabel(axMAE_AE, 'MAE (°C)', 'FontSize', custom_fontsize);
    title(axRatioAE, 'Autoencoder', 'FontSize', custom_fontsize+2, 'FontWeight', 'normal');
    %title(axMAE_AE, 'Autoencoder', 'FontSize', custom_fontsize+1, 'FontWeight', 'normal');

    % ---------------------------------------------------------------
    % Fila inferior: Rice-Golomb y Gorilla, x = input_dim (tamano de ventana)
    % ---------------------------------------------------------------
    cls_tables = {riceSweep, gorillaSweep};
    cls_colors = {COLOR_RICE, COLOR_GORILLA};
    cls_specs  = {'-p', '--h'};   % pentagram para Rice, hexagram para Gorilla

    for i = 1:numel(cls_tables)
        clsSweep = cls_tables{i};
        inputDims = sort(unique(clsSweep.input_dim));

        ratio_mean_v = zeros(numel(inputDims), 1); ratio_ci_v = zeros(numel(inputDims), 1);
        mae_mean_v   = zeros(numel(inputDims), 1); mae_ci_v   = zeros(numel(inputDims), 1);

        for k = 1:numel(inputDims)
            mask = clsSweep.input_dim == inputDims(k);

            [ratio_mean_v(k), ratio_ci_v(k)] = ConfidenceInterval(clsSweep.ratio(mask));
            [mae_mean_v(k), mae_ci_v(k)]     = ConfidenceInterval(clsSweep.mae(mask));
        end

        h = errorbar(axRatioCls, inputDims, ratio_mean_v, ratio_ci_v, cls_specs{i}, 'Color', cls_colors{i}, 'MarkerFaceColor', cls_colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        txt_initial = sprintf('(%d, %.2f)', inputDims(1), ratio_mean_v(1));
        txt_final   = sprintf('(%d, %.2f)', inputDims(end), ratio_mean_v(end));
        text(axRatioCls, inputDims(1), ratio_mean_v(1), txt_initial, 'Color', cls_colors{i}, 'FontSize', 6, 'VerticalAlignment', 'bottom');
        text(axRatioCls, inputDims(end), ratio_mean_v(end), txt_final, 'Color', cls_colors{i}, 'FontSize', 6, 'VerticalAlignment', 'bottom');
        errorbar(axMAE_Cls, inputDims, mae_mean_v, mae_ci_v, cls_specs{i}, 'Color', cls_colors{i}, 'MarkerFaceColor', cls_colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(2 + i) = h;
    end

    % set y limits equal to AE 
    ae_ylim = ylim(axMAE_AE);
    %ylim(axMAE_AE, [0, ae_ylim(2)]);
    ylim(axMAE_Cls, [0, ae_ylim(2)]);
    %linkaxes([axMAE_AE, axMAE_Cls], 'y');
    % Comment xlabel because is covered bu axMAE_Cls and ylabels because
    % are covered by AEs labels
    %xlabel(axRatioCls, 'Input dimension (window size)'); 
    %ylabel(axRatioCls, 'Compression ratio');
    xlabel(axMAE_Cls, 'Window size', 'FontSize', custom_fontsize);  
    %ylabel(axMAE_Cls, 'MAE (°C) \times10^{-6}');
    title(axRatioCls, 'Rice-Golomb / Gorilla', 'FontSize', custom_fontsize+2, 'FontWeight', 'normal');
    %title(axMAE_Cls, 'Rice-Golomb / Gorilla', 'FontSize', custom_fontsize+1, 'FontWeight', 'normal');

    hold(axRatioAE, 'off'); hold(axMAE_AE, 'off');
    hold(axRatioCls, 'off'); hold(axMAE_Cls, 'off');

    set(axRatioAE, 'FontSize', custom_fontsize);
    set(axMAE_AE, 'FontSize', custom_fontsize);
    set(axRatioCls, 'FontSize', custom_fontsize);
    set(axMAE_Cls, 'FontSize', custom_fontsize);

    % Leyenda global fuera, abajo
    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal', 'FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', numel(legend_entries));
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    % Export file to PDF
    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
