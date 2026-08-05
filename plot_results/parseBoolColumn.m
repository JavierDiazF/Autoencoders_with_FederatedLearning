function boolCol = parseBoolColumn(col)
    % Convert strings "True"/"False", from table to boolean
    if islogical(col)
        boolCol = col;
    else
        boolCol = string(col) == "True";
    end
end