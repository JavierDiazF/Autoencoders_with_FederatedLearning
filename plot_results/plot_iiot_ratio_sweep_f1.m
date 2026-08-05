function plot_iiot_ratio_sweep_f1(sweep, file_name)
    % plot_iiot_ratio_sweep_f1(sweep, file_name)
    %
    % F1 vs target compression ratio for the ratio sweep, split into
    % symmetric (left) / asymmetric (right) panels, one colored line per
    % input_dim (window size) -- same convention as plot_sweep_mae.m
    % (dimColors = lines(n), sym/asym as separate panels).

    col_width = 3.5;
    fig_height = 2.35;
    fig = figure('Units', 'inches', 'Position', [1 1 col_width fig_height]);
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [col_width fig_height]);
    set(fig, 'PaperPosition', [0 0 col_width fig_height]);
    set(fig, 'PaperPositionMode', 'manual');

    custom_fontsize = 8;

    sweep = sweep(string(sweep.scenario) == "latent", :);
    inputDims = sort(unique(sweep.input_dim));
    dimColors = lines(numel(inputDims));

    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    axSym  = nexttile; hold(axSym, 'on');
    axAsym = nexttile; hold(axAsym, 'on');
    sgtitle(t, 'F1 vs compression ratio, by window size', ...
        'FontSize', custom_fontsize + 1, 'FontWeight', 'normal');

    legend_handles = gobjects(1, numel(inputDims));
    legend_entries = cell(1, numel(inputDims));

    for d = 1:numel(inputDims)
        dimVal = inputDims(d);
        maskDim = sweep.input_dim == dimVal;

        ratios = sort(unique(sweep.target_ratio(maskDim)));

        f1sym_mean = zeros(numel(ratios), 1); f1sym_ci = zeros(numel(ratios), 1);
        f1asym_mean = zeros(numel(ratios), 1); f1asym_ci = zeros(numel(ratios), 1);

        for k = 1:numel(ratios)
            maskSym  = maskDim & sweep.symmetric == true  & sweep.target_ratio == ratios(k);
            maskAsym = maskDim & sweep.symmetric == false & sweep.target_ratio == ratios(k);
            [f1sym_mean(k), f1sym_ci(k)]   = ConfidenceInterval(sweep.f1(maskSym));
            [f1asym_mean(k), f1asym_ci(k)] = ConfidenceInterval(sweep.f1(maskAsym));
        end

        h = errorbar(axSym, ratios, f1sym_mean, f1sym_ci, '-o', ...
            'Color', dimColors(d,:), 'MarkerFaceColor', dimColors(d,:), 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axAsym, ratios, f1asym_mean, f1asym_ci, '-o', ...
            'Color', dimColors(d,:), 'MarkerFaceColor', dimColors(d,:), 'LineWidth', 1.2, 'MarkerSize', 4);

        legend_handles(d) = h;
        legend_entries{d} = sprintf('w=%d', dimVal);
    end

    hold(axSym, 'off'); hold(axAsym, 'off');

    xlabel(axSym, sprintf('Compression ratio\n(Symmetric AE)')); ylabel(axSym, 'F1 score');
    xlabel(axAsym, sprintf('Compression ratio\n(Asymmetric AE)')); ylabel(axAsym, 'F1 score');
    set(axSym, 'FontSize', custom_fontsize);
    set(axAsym, 'FontSize', custom_fontsize);
    ylim(axSym, [0, 1]); ylim(axAsym, [0, 1]);

    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal', ...
        'FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', ceil(numel(inputDims) / 2));
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
