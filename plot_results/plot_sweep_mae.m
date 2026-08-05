function plot_sweep_mae(sweep, file_name)
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

    % Initialize plot values
    inputDims = unique(sweep.input_dim);
    dimColors = lines(numel(inputDims));
    
    % Get stats (mean and std)
    %summary = groupsummary(sweep, {'input_dim', 'latent_dim', 'symmetric'}, {'mean', 'std'}, {'mse_mean', 'mae_mean'});
    %cauclluated_std = ConfidenceInterval(summary.GroupCount)
    % En realidad, para calcular el ratio se deberian tener en cuenta 
    % los 3 bytes de ref, min y max necesarios para la desnormalizacion
    % Por ahora no los tengo en cuenta porque quiero ver cómo queda
    %summary.ratio = summary.input_dim ./ summary.latent_dim;
    
    % Tiled layout MSE left, MAE right
    t = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');
    axMAEsym = nexttile; hold(axMAEsym, 'on');
    axMAEasym = nexttile; hold(axMAEasym, 'on');
    % Global title
    sgtitle(t, 'MAE vs ratio compression', 'FontSize',custom_fontsize + 1, 'FontWeight','normal');
    
    % Manage global legend
    legend_handles = gobjects(1, numel(inputDims)*2);
    legend_entries = cell(1, numel(inputDims)*2);
    idx = 0;
    % For loop
    for inputdim_idx = 1:numel(inputDims)
        for symVal = [true, false]
            latentDims = unique(sweep.latent_dim(sweep.input_dim == inputDims(inputdim_idx) & sweep.symmetric == symVal));
            latentDims = sort(latentDims);

            ratios = zeros(numel(latentDims), 1);
            %mse_mean_v = zeros(numel(latentDims), 1); mse_ci_v = zeros(numel(latentDims), 1);
            mae_mean_v = zeros(numel(latentDims), 1); mae_ci_v = zeros(numel(latentDims), 1);
            for k = 1:numel(latentDims)
                mask = sweep.input_dim == inputDims(inputdim_idx) & sweep.symmetric == symVal & sweep.latent_dim == latentDims(k);
    
                %mse_vals = sweep.mse_mean(mask);   % los 10 valores (uno por semilla) de esta config exacta
                mae_vals = sweep.mae_mean(mask);
    
                %[mse_mean_v(k), mse_ci_v(k)] = ConfidenceInterval(mse_vals);
                [mae_mean_v(k), mae_ci_v(k)] = ConfidenceInterval(mae_vals);
                ratios(k) = inputDims(inputdim_idx) / latentDims(k);
            end
            if symVal
                lineSpec = '-o';
                h = errorbar(axMAEsym, ratios, mae_mean_v, mae_ci_v, lineSpec, 'Color', dimColors(inputdim_idx,:), 'MarkerFaceColor', dimColors(inputdim_idx,:), 'LineWidth', 1.2, 'MarkerSize', 4);
            else
                lineSpec = '--s';
                errorbar(axMAEasym, ratios, mae_mean_v, mae_ci_v, lineSpec, 'Color', dimColors(inputdim_idx,:), 'MarkerFaceColor', dimColors(inputdim_idx,:), 'LineWidth', 1.2, 'MarkerSize', 4);
            end
        end
        idx = idx +1;
        legend_handles(idx) = h;
        legend_entries{idx} = sprintf('=%d', inputDims(inputdim_idx));
    end
    hold(axMAEsym, 'off'); hold(axMAEasym, 'off');
    xlabel(axMAEsym, sprintf('Compression ratio\n(Symmetric AE)'), 'FontSize',custom_fontsize - 1); ylabel(axMAEsym, 'MAE (°C)', 'FontSize',custom_fontsize - 1);
    xlabel(axMAEasym, sprintf('Compression ratio\n(Asymmetric AE)', 'FontSize',custom_fontsize - 1)); ylabel(axMAEasym, 'MAE (°C)', 'FontSize',custom_fontsize - 1);
    set(axMAEsym, 'FontSize', custom_fontsize);
    set(axMAEasym, 'FontSize', custom_fontsize);
    % Leyenda global fuera, abajo
    lgd = legend(legend_handles(1:idx), legend_entries(1:idx), 'Orientation', 'horizontal', 'FontSize',custom_fontsize, 'Box','off', 'NumColumns',idx);
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    % Export file to PDF
    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig,full_file_name,'ContentType','vector');
end