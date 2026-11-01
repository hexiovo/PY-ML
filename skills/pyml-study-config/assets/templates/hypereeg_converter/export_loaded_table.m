function export_loaded_table(cfg,loaded)
% Native MATLAB decoding: no guessed HDF5 table/string/cell reconstruction.
    selected = string(cfg.table_path);
    if strlength(selected) == 0
        if isfield(loaded,'FeatureResult') && isfield(loaded.FeatureResult,'longData')
            selected = "FeatureResult.longData";
        else
            paths = collectTables(loaded, "");
            if numel(paths) ~= 1
                error('PYML:MappingRequired', ...
                    '请填写一个明确的 MAT table 路径；可用表：%s。任意数组不会自动转置/展平。', strjoin(paths, ', '));
            end
            selected = paths(1);
        end
    end
    if ~startsWith(selected,["FeatureResult.","Artifact.payload", ...
            "Results.firstOrder.","Results.secondOrder."])
        error('PYML:InvalidRoot','只允许正式 FeatureResult.longData 或 Results/Artifact 数值载荷表。');
    end
    parts = split(selected,'.');
    data = loaded;
    for i = 1:numel(parts)
        name = char(parts(i));
        if ~isstruct(data) || ~isscalar(data) || ~isfield(data,name)
            error('PYML:InvalidPath','不是明确的标量结构路径：%s',selected);
        end
        data = data.(name);
    end
    if ~istable(data)
        error('PYML:NotTable','目标不是 MATLAB table，不能任意转置/展平。');
    end
    classes = strings(1,width(data));
    units = data.Properties.VariableUnits;
    names = data.Properties.VariableNames;
    for i = 1:width(data)
        column = data.(names{i});
        classes(i) = string(class(column));
        if size(column,2) ~= 1 || ndims(column) ~= 2 || ...
                ~(isnumeric(column) || islogical(column) || isstring(column) || ...
                  iscellstr(column) || iscategorical(column)) || ...
                (isnumeric(column) && ~isreal(column))
            error('PYML:UnsupportedColumn', ...
                '列 %s 为 %s 或多维/复数值；需要源项目明确坐标映射。',names{i},class(column));
        end
    end
    output = fullfile(cfg.output,'table.csv');
    file = fopen(output,'w','n','UTF-8');
    if file < 0, error('PYML:Output','不能创建输出。'); end
    cleanup = onCleanup(@() fclose(file));
    for i = 1:numel(names)
        if i > 1, fprintf(file,','); end
        fprintf(file,'%s',csvQuote(string(names{i})));
    end
    fprintf(file,'\n');
    for r = 1:height(data)
        for c = 1:width(data)
            if c > 1, fprintf(file,','); end
            value = data.(names{c})(r,:);
            if isfloat(value)
                if isnan(value), fprintf(file,'NaN');
                else, fprintf(file,'%.17g',value); end
            elseif isinteger(value)
                if startsWith(class(value),'uint'), fprintf(file,'%u',value);
                else, fprintf(file,'%d',value); end
            elseif islogical(value)
                if value, fprintf(file,'true'); else, fprintf(file,'false'); end
            else
                value = string(value);
                if ismissing(value), value = ""; end
                fprintf(file,'%s',csvQuote(value));
            end
        end
        fprintf(file,'\n');
    end
    clear cleanup
    info = struct('path',selected,'rows',height(data),'columns',{names}, ...
        'classes',classes,'units',{units},'matlab_version',string(version), ...
        'dimension_policy',"native table rows; scalar columns; no transpose or flatten");
    if isfield(loaded,'Artifact')
        a = loaded.Artifact;
        keep = {'artifactId','taskId','producerId','producerVersion','scope','scopeKey', ...
            'sourceIds','scientificSignature','quality','audit'};
        for i = 1:numel(keep)
            if isfield(a,keep{i}), info.(keep{i}) = a.(keep{i}); end
        end
    end
    file = fopen(fullfile(cfg.output,'table.json'),'w','n','UTF-8');
    if file < 0, error('PYML:Output','不能创建来源记录。'); end
    cleanup = onCleanup(@() fclose(file));
    fprintf(file,'%s',jsonencode(info));
end

function paths = collectTables(value, prefix)
    paths = strings(0,1);
    if istable(value)
        paths = prefix;
    elseif isstruct(value) && isscalar(value)
        names = fieldnames(value);
        for i = 1:numel(names)
            key = string(names{i});
            if strlength(prefix) > 0, key = prefix + "." + key; end
            paths = [paths; collectTables(value.(names{i}),key)]; %#ok<AGROW>
        end
    end
end

function output = csvQuote(value)
    output = '"' + replace(value,'"','""') + '"';
end
