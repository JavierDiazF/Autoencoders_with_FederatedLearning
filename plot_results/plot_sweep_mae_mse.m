function plot_sweep_mae_mse(sweep, file_name)
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
    summary = groupsummary(sweep, {'input_dim', 'latent_dim', 'symmetric'}, {'mean', 'std'}, {'mse_mean', 'mae_mean'});
    %cauclluated_std = ConfidenceInterval(summary.GroupCount)
    % En realidad, para calcular el ratio se deberian tener en cuenta 
    % los 3 bytes de ref, min y max necesarios para la desnormalizacion
    % Por ahora no los tengo en cuenta porque quiero ver cómo queda
    summary.ratio = summary.input_dim ./ summary.latent_dim;
    
    % Tiled layout MSE left, MAE right
    t = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');
    axMSE = nexttile; hold(axMSE, 'on');
    axMAE = nexttile; hold(axMAE, 'on');
    % Global title
    sgtitle(t, 'MSE and MAE vs ratio compression', 'FontSize',custom_fontsize, 'FontWeight','normal');
    
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
            mse_mean_v = zeros(numel(latentDims), 1); mse_ci_v = zeros(numel(latentDims), 1);
            mae_mean_v = zeros(numel(latentDims), 1); mae_ci_v = zeros(numel(latentDims), 1);
            for k = 1:numel(latentDims)
                mask = sweep.input_dim == inputDims(inputdim_idx) & sweep.symmetric == symVal & sweep.latent_dim == latentDims(k);
    
                mse_vals = sweep.mse_mean(mask);   % los 10 valores (uno por semilla) de esta config exacta
                mae_vals = sweep.mae_mean(mask);
    
                [mse_mean_v(k), mse_ci_v(k)] = ConfidenceInterval(mse_vals);
                [mae_mean_v(k), mae_ci_v(k)] = ConfidenceInterval(mae_vals);
                ratios(k) = inputDims(inputdim_idx) / latentDims(k);
            end
            if symVal
                lineSpec = '-o'; labelSuffix = 'Symm.';
            else
                lineSpec = '--s'; labelSuffix = 'Asymm';
            end
            h = errorbar(axMSE, ratios, mse_mean_v, mse_ci_v, lineSpec, 'Color', dimColors(inputdim_idx,:), 'MarkerFaceColor', dimColors(inputdim_idx,:), 'LineWidth', 1.2, 'MarkerSize', 4);
            errorbar(axMAE, ratios, mae_mean_v, mae_ci_v, lineSpec, 'Color', dimColors(inputdim_idx,:), 'MarkerFaceColor', dimColors(inputdim_idx,:), 'LineWidth', 1.2, 'MarkerSize', 4);

            % % Filter summary data
            % subset = summary(summary.input_dim == inputDims(inputdim_idx) & summary.symmetric == symVal, :)
            % 
            % subset = sortrows(subset, 'ratio');
            % if isempty(subset)
            %     continue;
            % end
            % if symVal
            %     lineSpec = '-o'; labelSuffix = 'Symm.';
            % else
            %     lineSpec = '--s'; labelSuffix = 'Asymm';
            % end
            % % Get subset mse
            % h = errorbar(axMSE, subset.ratio, subset.mean_mse_mean, subset.std_mse_mean, lineSpec, 'Color', dimColors(inputdim_idx, :), 'MarkerFaceColor',dimColors(inputdim_idx, :), 'LineWidth', 1.2, 'MarkerSize', 4);
            % errorbar(axMAE, subset.ratio, subset.mean_mae_mean, subset.std_mae_mean, lineSpec, 'Color', dimColors(inputdim_idx, :), 'MarkerFaceColor',dimColors(inputdim_idx, :), 'LineWidth', 1.2, 'MarkerSize', 4);
            
            idx = idx +1;
            legend_handles(idx) = h;
            legend_entries{idx} = sprintf('=%d, %s', inputDims(inputdim_idx), labelSuffix);
        end
    end
    hold(axMSE, 'off'); hold(axMAE, 'off');
    xlabel(axMSE, 'Compression ratio'); ylabel(axMSE, 'MSE (°C^2)');
    xlabel(axMAE, 'Compression ratio'); ylabel(axMAE, 'MAE (°C)');
    set(axMSE, 'FontSize', custom_fontsize);
    set(axMAE, 'FontSize', custom_fontsize);
    % Leyenda global fuera, abajo
    lgd = legend(legend_handles(1:idx), legend_entries(1:idx), 'Orientation', 'horizontal', 'FontSize',custom_fontsize, 'Box','off', 'NumColumns',ceil(idx/2));
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [8 6];

    % Export file to PDF
    full_file_name = strcat(file_name, '.pdf');
    exportgraphics(fig,full_file_name,'ContentType','vector');
end