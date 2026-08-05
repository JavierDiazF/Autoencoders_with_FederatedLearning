function [media,intervalo] = ConfidenceInterval(medias)
%% Función para calcular el intervalo de confianza
% calcula el intervalo de confianza y la media de medias
% para distintas tiradas de un mismo eperimento con una garantia del 0.975
% - input:
%     - medias: datos de las distintas tiradas para el mismo experimento
% - output:
%     - media: media de medias de la tirada
%     - intervalo: intervalo de confianza
    num_repeticiones=length(medias);
    grados_libertad=(num_repeticiones-1);
    tstudent=(tinv([0.025 0.975],grados_libertad));
    media=mean(medias);
    intervalo = (tstudent(2).* std(medias))./sqrt(grados_libertad);    
end