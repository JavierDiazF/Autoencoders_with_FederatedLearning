function plot_iiot_latent_sweep(sweep, file_name)
    % plot_iiot_latent_sweep(sweep, file_name)
    %
    % 2-panel figure for the IIoT latent-dim sweep (fixed input_dim, latent_dim
    % swept): left = payload size sent to BentoML, right = F1 of the BentoML
    % inference, both vs latent_dim, with raw / symmetric AE / asymmetric AE
    % overlaid. Raw doesn't depend on latent_dim (it never compresses), so
    % it's drawn as a flat reference line + shaded CI band across the same
    % x-range as the AE curves, rather than an errorbar series.

    col_width = 3.5;
    fig_height = 2.35;
    fig = figure('Units', 'inches', 'Position', [1 1 col_width fig_height]);
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [col_width fig_height]);
    set(fig, 'PaperPosition', [0 0 col_width fig_height]);
    set(fig, 'PaperPositionMode', 'manual');

    custom_fontsize = 8;
    COLOR_SYM  = [0.00, 0.45, 0.74];   % azul  -> simetrico
    COLOR_ASYM = [0.85, 0.33, 0.10];   % naranja -> asimetrico
    COLOR_RAW  = [0.30, 0.30, 0.30];   % gris  -> raw (referencia sin compresion)

    scenario = string(sweep.scenario);
    isRaw    = scenario == "raw";
    isLatent = scenario == "latent";

    latentDims = sort(unique(sweep.latent_dim(isLatent)));
    xrange = [min(latentDims), max(latentDims)];

    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    axPayload = nexttile; hold(axPayload, 'on');
    axF1      = nexttile; hold(axF1, 'on');

    input_dim = sweep.input_dim(1);
    sgtitle(t, sprintf('Payload size and F1 vs latent dimension (input\\_dim=%d)', input_dim), ...
        'FontSize', custom_fontsize + 1, 'FontWeight', 'normal', 'Interpreter', 'tex');

    legend_handles = gobjects(1, 3);
    legend_entries = {'Raw', 'Symmetric', 'Asymmetric'};

    % --- raw: flat reference line + shaded CI band (doesn't vary with latent_dim) ---
    [payload_raw_mean, payload_raw_ci] = ConfidenceInterval(sweep.payload_bytes_theoretical(isRaw));
    [f1_raw_mean, f1_raw_ci]           = ConfidenceInterval(sweep.f1(isRaw));

    fill(axPayload, [xrange, fliplr(xrange)], ...
        [payload_raw_mean - payload_raw_ci, payload_raw_mean - payload_raw_ci, ...
         payload_raw_mean + payload_raw_ci, payload_raw_mean + payload_raw_ci], ...
        COLOR_RAW, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    h = plot(axPayload, xrange, [payload_raw_mean, payload_raw_mean], ':', 'Color', COLOR_RAW, 'LineWidth', 1.2);
    legend_handles(1) = h;

    fill(axF1, [xrange, fliplr(xrange)], ...
        [f1_raw_mean - f1_raw_ci, f1_raw_mean - f1_raw_ci, ...
         f1_raw_mean + f1_raw_ci, f1_raw_mean + f1_raw_ci], ...
        COLOR_RAW, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(axF1, xrange, [f1_raw_mean, f1_raw_mean], ':', 'Color', COLOR_RAW, 'LineWidth', 1.2);

    % --- symmetric / asymmetric AE, one errorbar series each ---
    labels = [true, false];
    colors = {COLOR_SYM, COLOR_ASYM};

    for i = 1:numel(labels)
        symVal = labels(i);

        payload_mean_v = zeros(numel(latentDims), 1); payload_ci_v = zeros(numel(latentDims), 1);
        f1_mean_v      = zeros(numel(latentDims), 1); f1_ci_v      = zeros(numel(latentDims), 1);

        for k = 1:numel(latentDims)
            mask = isLatent & sweep.symmetric == symVal & sweep.latent_dim == latentDims(k);

            [payload_mean_v(k), payload_ci_v(k)] = ConfidenceInterval(sweep.payload_bytes_theoretical(mask));
            [f1_mean_v(k), f1_ci_v(k)]           = ConfidenceInterval(sweep.f1(mask));
        end

        if symVal
            lineSpec = '-o';
        else
            lineSpec = '--s';
        end

        h = errorbar(axPayload, latentDims, payload_mean_v, payload_ci_v, lineSpec, ...
            'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        errorbar(axF1, latentDims, f1_mean_v, f1_ci_v, lineSpec, ...
            'Color', colors{i}, 'MarkerFaceColor', colors{i}, 'LineWidth', 1.2, 'MarkerSize', 4);
        legend_handles(i + 1) = h;
    end

    hold(axPayload, 'off'); hold(axF1, 'off');

    xlabel(axPayload, 'Latent dimension'); ylabel(axPayload, 'Payload size (bytes)');
    xlabel(axF1, 'Latent dimension'); ylabel(axF1, 'F1 score');
    set(axPayload, 'FontSize', custom_fontsize);
    set(axF1, 'FontSize', custom_fontsize);
    xlim(axPayload, xrange); xlim(axF1, xrange);
    ylim(axF1, [0, 1]);

    lgd = legend(legend_handles, legend_entries, 'Orientation', 'horizontal', ...
        'FontSize', custom_fontsize, 'Box', 'off', 'NumColumns', 3);
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig, full_file_name, 'ContentType', 'vector');
end
