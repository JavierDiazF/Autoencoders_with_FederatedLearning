function plot_sweep_hidden_layers_triangle(sweep, filename)
    % Como plot_sweep_hidden_layers pero juntando las tres gráficas
    %   arriba-izda: energia del encoder vs hidden layers
    %   arriba-dcha: energia del decoder vs hidden layers
    %   abajo: MAE

    % Size parameters for plot
    col_width = 3.5;
    %fig_height = 4.4;
    fig_height = 3;
    fig = figure('Units','inches','Position',[1 1 col_width fig_height]);
    set(fig,'PaperUnits','inches');
    set(fig,'PaperSize',[col_width fig_height]);
    set(fig,'PaperPosition',[0 0 col_width fig_height]);
    set(fig,'PaperPositionMode','manual');

    % Font size
    custom_fontsize = 8;
    ENERGY_PJ_PER_MAC = 4.6;   % Horowitz, ISSCC 2014, 45nm float32 mult+add (pJ/MAC)
    COLOR_SYM  = [0.00, 0.45, 0.74];
    COLOR_ASYM = [0.85, 0.33, 0.10];

    sweep.encoder_energy_nJ = sweep.encoder_macs * ENERGY_PJ_PER_MAC / 1000;
    sweep.decoder_energy_nJ = sweep.decoder_macs * ENERGY_PJ_PER_MAC / 1000;

    t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    %t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(t, sprintf('Encoder/decoder energy and MAE vs hidden layers'), 'FontSize', custom_fontsize+1, 'FontWeight', 'normal', 'Interpreter', 'tex');

    axEncoder = nexttile(1); hold(axEncoder, 'on');
    %axEncoder = nexttile(1, [2,1]); hold(axEncoder, 'on');
    axDecoder = nexttile(2); hold(axDecoder, 'on');
    %axDecoder = nexttile(2, [2,1]); hold(axDecoder, 'on');
    axMAE = nexttile(3, [1 2]); hold(axMAE, 'on');   % ocupa las 2 columnas de la fila de abajo
    %axMAE = nexttile(5, [1 2]); hold(axMAE, 'on');

    hiddenLayerVals = sort(unique(sweep.hidden_layers));

    legend_handles = gobjects(1, 2);
    legend_entries = {'Symmetric (AE)', 'Asymmetric (AAE)'};
    labels = [true, false];
    colors = {COLOR_SYM, COLOR_ASYM};

    for i = 1:numel(labels)
        symVal = labels(i);

        encMean_v = zeros(numel(hiddenLayerVals), 1); encCI_v = zeros(numel(hiddenLayerVals), 1);
        decMean_v = zeros(numel(hiddenLayerVals), 1); decCI_v = zeros(numel(hiddenLayerVals), 1);
        maeMean_v = zeros(numel(hiddenLayerVals), 1); maeCI_v = zeros(numel(hiddenLayerVals), 1);

        for k = 1:numel(hiddenLayerVals)
            mask = sweep.symmetric == symVal & sweep.hidden_layers == hiddenLayerVals(k);

            [encMean_v(k), encCI_v(k)] = ConfidenceInterval(sweep.encoder_energy_nJ(mask));
            [decMean_v(k), decCI_v(k)] = ConfidenceInterval(sweep.decoder_energy_nJ(mask));
            [maeMean_v(k), maeCI_v(k)] = ConfidenceInterval(sweep.mae_mean(mask));
        end

        h = errorbar(axEncoder, hiddenLayerVals, encMean_v, encCI_v, '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axDecoder, hiddenLayerVals, decMean_v, decCI_v, '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axMAE, hiddenLayerVals, maeMean_v, maeCI_v, '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i) = h;
    end
    hold(axEncoder, 'off'); hold(axDecoder, 'off'); hold(axMAE, 'off');

    tickLabels = arrayfun(@(x) sprintf('AE-%d\\newlineAAE-%d', x, x), hiddenLayerVals, 'UniformOutput', false);

    ylabel(axEncoder, 'Encoder energy (nJ)');
    ylabel(axDecoder, 'Decoder energy (nJ)');
    ylabel(axMAE, 'MAE (°C)');

    all_axes = [axEncoder, axDecoder, axMAE];
    for idx = 1:numel(all_axes)
        ax = all_axes(idx);
        xticks(ax, hiddenLayerVals);
        ax.XTickLabel = tickLabels;
        xtickangle(ax, 0);
        ax.TickLabelInterpreter = 'tex';
        set(ax, 'FontSize', custom_fontsize);
        ax.XAxis.FontSize = custom_fontsize - 2;
        ax.YAxis.FontSize = custom_fontsize - 2;
        if idx == numel(all_axes)
            xlim(ax, [min(hiddenLayerVals)-1, max(hiddenLayerVals)+1]);
        else
            xlim(ax, [min(hiddenLayerVals), max(hiddenLayerVals)]); 
        end
    end

    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal', 'FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', 2);
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    full_file_name = strcat(filename, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
