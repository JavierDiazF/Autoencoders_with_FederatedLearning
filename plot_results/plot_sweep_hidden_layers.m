function plot_sweep_hidden_layers(sweep, filename, mode)
    % plot_sweep_hidden_layers(sweep, filename, mode)
    % If not incuded mode it will show encoder o decoder
    if nargin < 3
        mode = 'encoder_decoder';
    end

    % Size parameters for plot
    col_width = 3.5;
    fig_height = 2.35;
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

    switch mode
        case 'encoder_decoder'
            leftVar  = 'encoder_energy_nJ'; leftLabel  = 'Encoder energy (nJ)';
            rightVar = 'decoder_energy_nJ'; rightLabel = 'Decoder energy (nJ)';
            figTitle = 'Encoder and decoder energy vs hidden layers';
        case 'mse_encoder'
            leftVar  = 'mse_mean';          leftLabel  = 'MSE (\circC^2)';
            rightVar = 'encoder_energy_nJ'; rightLabel = 'Encoder energy (nJ)';
            figTitle = 'MSE and encoder energy vs hidden layers';
        case 'mae_encoder'
            leftVar  = 'mae_mean';          leftLabel  = 'MAE (\circC)';
            rightVar = 'encoder_energy_nJ'; rightLabel = 'Encoder energy (nJ)';
            figTitle = 'MAE and encoder energy vs hidden layers';
        otherwise
            error('plot_sweep_hidden_layers:badMode','mode must be ''encoder_decoder'' or ''mse_encoder'', no ''%s''', mode);
    end

    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(t, figTitle, 'FontSize', custom_fontsize, 'FontWeight', 'normal');

    axLeft  = nexttile; hold(axLeft, 'on');
    axRight = nexttile; hold(axRight, 'on');

    hiddenLayerVals = sort(unique(sweep.hidden_layers));

    legend_handles = gobjects(1, 2);
    legend_entries = {'Symmetric (AE)', 'Asymmetric (AAE)'};
    labels = [true, false];
    colors = {COLOR_SYM, COLOR_ASYM};

    for i = 1:numel(labels)
        symVal = labels(i);

        leftMean_v  = zeros(numel(hiddenLayerVals), 1); leftCI_v  = zeros(numel(hiddenLayerVals), 1);
        rightMean_v = zeros(numel(hiddenLayerVals), 1); rightCI_v = zeros(numel(hiddenLayerVals), 1);

        for k = 1:numel(hiddenLayerVals)
            mask = sweep.symmetric == symVal & sweep.hidden_layers == hiddenLayerVals(k);

            leftVals  = sweep.(leftVar)(mask);
            rightVals = sweep.(rightVar)(mask);

            [leftMean_v(k), leftCI_v(k)]   = ConfidenceInterval(leftVals);
            [rightMean_v(k), rightCI_v(k)] = ConfidenceInterval(rightVals);
        end

        h = errorbar(axLeft, hiddenLayerVals, leftMean_v, leftCI_v, '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axRight, hiddenLayerVals, rightMean_v, rightCI_v, '-o', 'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i) = h;
    end
    hold(axLeft, 'off'); hold(axRight, 'off');

    % Create xticks
    %tickLabels = ['AE-1\newlineAAE-1'; 'AE-2\newlineAAE-2'; 'AE-3\newlineAAE-3'; 'AE-4\newlineAAE-4'; 'AE-5\newlineAAE-5'];
    tickLabels = arrayfun(@(x) sprintf('AE-%d\\newlineAAE-%d', x, x), hiddenLayerVals, 'UniformOutput', false);
    %tickLabels = arrayfun(@(x) {sprintf('AE-%d', x), sprintf('AAE-%d', x)}, hiddenLayerVals, 'UniformOutput', false);
    %disp(tickLabels)
    %xlabel(axLeft, 'Type of Autoencoder'); 
    ylabel(axLeft, leftLabel);
    %xlabel(axRight, 'Type of Autoencoder'); 
    %ylabel(axRight, rightLabel);
    xticks(axLeft, hiddenLayerVals);  axLeft.XTickLabel = tickLabels; xtickangle(axLeft, 0);
    xticks(axRight, hiddenLayerVals);  axRight.XTickLabel = tickLabels; xtickangle(axRight, 0);
    axLeft.TickLabelInterpreter = 'tex';
    axRight.TickLabelInterpreter = 'tex';
    set(axLeft, 'FontSize', custom_fontsize);
    set(axRight, 'FontSize', custom_fontsize);

    axLeft.XAxis.FontSize = custom_fontsize - 2;
    axRight.XAxis.FontSize = custom_fontsize - 2;

    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal','FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', 2);
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    full_file_name = strcat(filename, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
