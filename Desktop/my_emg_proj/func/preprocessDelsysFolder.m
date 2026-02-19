function out = preprocessDelsysFolder(folder, opts)
% preprocessDelsysFolder
% 批量读取Delsys导出的csv（有表头），取奇数列为时间戳、偶数列为EMG；
% 对EMG做去均值、带通滤波，并可自动判定是否需要50Hz陷波。
%
% 输入:
%   folder (char/string): csv所在文件夹
%   opts: 结构体参数（�� run_main.m 示例）
%
% 输出:
%   out: 结构体，包含各文件的时间、原始EMG、带通后EMG、最终EMG等


if nargin < 2
    error('preprocessDelsysFolder requires 2 inputs: folder and opts.');
end

% 兼容 folder 为 char / string
folder = char(folder);

% 给 opts 填默认值（缺哪个补哪个）
if ~isfield(opts,'fs'), opts.fs = 1926; end
if ~isfield(opts,'timeCols'), opts.timeCols = [1 3 5 7 9 11]; end
if ~isfield(opts,'emgCols'),  opts.emgCols  = [2 4 6 8 10 12]; end
if ~isfield(opts,'bandpass'), opts.bandpass = [20 450]; end
if ~isfield(opts,'bpOrder'),  opts.bpOrder  = 4; end

if ~isfield(opts,'enableNotchAuto'), opts.enableNotchAuto = true; end
if ~isfield(opts,'notchFreq'), opts.notchFreq = 50; end
if ~isfield(opts,'notchQ'), opts.notchQ = 35; end
if ~isfield(opts,'notchThreshold_dB'), opts.notchThreshold_dB = 8; end
if ~isfield(opts,'notchMajorityChannels'), opts.notchMajorityChannels = 3; end

files = dir(fullfile(folder, "*.csv"));
if isempty(files)
    error("在文件夹中没有找到csv: %s", folder);
end

fs = opts.fs;

% 设计带通
[b_band, a_band] = butter(opts.bpOrder, opts.bandpass/(fs/2), 'bandpass');

% 设计陷波（50Hz）
wo = opts.notchFreq/(fs/2);
bw = wo/opts.notchQ;
[b_notch, a_notch] = iirnotch(wo, bw);

% 预分配输出
out = struct();
out.folder = folder;
out.files = files;

n = numel(files);
out.allTime    = cell(n,1);
out.allEmgRaw  = cell(n,1);
out.allEmgBP   = cell(n,1);
out.allEmgFilt = cell(n,1);

% 记录判据
out.notchRdb = nan(n, numel(opts.emgCols));
out.needNotch = false(n,1);

for i = 1:n
    filename = fullfile(folder, files(i).name);

    T = readtable(filename, 'VariableNamingRule','preserve'); % 保留原始表头命名，避免警告
    data = table2array(T);

    % 基本检查
    if size(data,2) < max([opts.timeCols opts.emgCols])
        error("文件 %s 列数不足。当前列数=%d", files(i).name, size(data,2));
    end

    x = data(:, opts.emgCols);
    if any(~isfinite(x(:)))
        warning("文件 %s EMG列存在 NaN/Inf，filtfilt 可能失败。", files(i).name);
        % 对每个通道插值补齐缺失（推荐）
        x = fillmissing(x,'linear','EndValues','nearest');

        % 如果还有Inf（极少见），可以再做一次处理
        x(~isfinite(x)) = 0;
    end

    t = data(:, opts.timeCols);

    % 去均值（每通道）
    x0 = x - mean(x, 1);

    % 带通
    x_bp = filtfilt(b_band, a_band, x0);

    % 自动判定是否需要陷波
    if opts.enableNotchAuto
        [needNotch, Rdb_ch] = needNotch50Hz(x_bp, fs, ...
            opts.notchFreq, opts.notchThreshold_dB, opts.notchMajorityChannels);

        out.needNotch(i) = needNotch;
        out.notchRdb(i,:) = Rdb_ch;

        if needNotch
            x_f = filtfilt(b_notch, a_notch, x_bp);
        else
            x_f = x_bp;
        end
    else
        x_f = x_bp;
    end

    out.allTime{i} = t;
    out.allEmgRaw{i} = x;
    out.allEmgBP{i} = x_bp;
    out.allEmgFilt{i} = x_f;

    % 可选：打印
    fprintf("%s | needNotch=%d | R_dB=[%s]\n", files(i).name, out.needNotch(i), num2str(out.notchRdb(i,:), '%.1f '));
end

end