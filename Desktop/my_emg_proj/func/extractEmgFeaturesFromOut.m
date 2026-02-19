function feat = extractEmgFeaturesFromOut(out, opts)
% extractEmgFeaturesFromOut
% ============================================================
% 从 preprocessDelsysFolder 的输出 out 中，对每个文件的 EMG（N×C）
% 进行滑窗切分，并提取常用时域/频域特征，最终生成特征矩阵 X。
%
% 输入
%   out : 结构体，至少需要 out.allEmgFilt 和 out.files
%         out.allEmgFilt{i} -> N×C double（预处理后的EMG）
%   opts: 结构体，特征提取参数（缺省字段会自动补齐）
%       opts.fs        : 采样率(Hz)，例如 1926
%       opts.win_ms    : 窗口长度(ms)，例如 200
%       opts.step_ms   : 步长(ms)，例如 50
%       opts.useChannels : 使用哪些通道，默认 1:size(x,2)
%
%       opts.features.time : 时域特征列表（cellstr）
%           例：{'MAV','RMS','WL','ZC','SSC'}
%       opts.features.freq : 频域特征列表（cellstr）
%           例：{'MNF','MDF','TTP'}  % TTP=总谱功率（Total Power）
%
%       opts.zcThreshold  : 过零阈值（避免噪声引起假过零），默认 0
%       opts.sscThreshold : SSC阈值，默认 0
%       opts.nfft         : 频谱计算的 NFFT（可选，默认 nextpow2(winSamples)）
%
% 输出
%   feat: 结构体
%       feat.X            : (总窗口数) × (特征维数) 的 double 矩阵
%       feat.featureNames : 1×D cell，每列对应一个特征名（含通道信息）
%       feat.fileIndex    : (总窗口数)×1，每行来自哪个文件
%       feat.windowIndex  : (总窗口数)×1，该文件的第几个窗口
%       feat.sampleStart  : (总窗口数)×1，窗口起点样本索引（1-based）
%       feat.sampleEnd    : (总窗口数)×1，窗口终点样本索引
%
% 说明
%   - 每个窗口按通道分别提特征，然后按 [ch1特征 ch2特征 ...] 拼接成一行
%   - 后续你可以在此基础上加入标签解析（从文件名/标注表生成 y）
% ============================================================

% -------- 参数默认值 --------
if ~isfield(opts,'fs'), opts.fs = 1926; end
if ~isfield(opts,'win_ms'), opts.win_ms = 200; end
if ~isfield(opts,'step_ms'), opts.step_ms = 50; end
if ~isfield(opts,'trimStart_s'), opts.trimStart_s = 0; end % 默认不裁剪开头
if ~isfield(opts,'trimEnd_s'),   opts.trimEnd_s   = 0; end % 默认不裁剪结尾

if ~isfield(opts,'features') || ~isstruct(opts.features)
    opts.features = struct();
end
if ~isfield(opts.features,'time')
    opts.features.time = {'MAV','RMS','WL','ZC','SSC'};
end
if ~isfield(opts.features,'freq')
    opts.features.freq = {'MNF','MDF'};
end

if ~isfield(opts,'zcThreshold'), opts.zcThreshold = 0; end
if ~isfield(opts,'sscThreshold'), opts.sscThreshold = 0; end

% -------- 基本检查 --------
if ~isfield(out,'allEmgFilt')
    error('out must contain out.allEmgFilt');
end

fs = opts.fs;
winS  = round(opts.win_ms/1000 * fs);
stepS = round(opts.step_ms/1000 * fs);
if winS <= 1 || stepS <= 0
    error('Invalid window/step: winS=%d, stepS=%d', winS, stepS);
end

% -------- 通道选择（默认所有通道）--------
firstX = out.allEmgFilt{1};
if ~isfield(opts,'useChannels') || isempty(opts.useChannels)
    opts.useChannels = 1:size(firstX,2);
end

timeList = opts.features.time;
freqList = opts.features.freq;

% -------- 先生成 featureNames（根据通道数与特征列表）--------
featureNames = {};
for ch = opts.useChannels
    for k = 1:numel(timeList)
        featureNames{end+1} = sprintf('ch%d_%s', ch, timeList{k}); %#ok<AGROW>
    end
    for k = 1:numel(freqList)
        featureNames{end+1} = sprintf('ch%d_%s', ch, freqList{k}); %#ok<AGROW>
    end
end
D = numel(featureNames);

% -------- 预计算总窗口数（便于一次性预分配矩阵）--------
trimStartS = round(opts.trimStart_s * fs); % 开头要裁剪的样本点数
trimEndS   = round(opts.trimEnd_s   * fs); % 结尾要裁剪的样本点数
nFiles = numel(out.allEmgFilt);
winsPerFile = zeros(nFiles,1);
for i = 1:nFiles
    x = out.allEmgFilt{i};
    N0 = size(x,1);

    % 计算裁剪后有效区间（1-based）
    sA = trimStartS + 1;   % 有效数据起点
    sB = N0 - trimEndS;    % 有效数据终点

    % 裁剪后没数据就跳过
    if sB <= sA
        winsPerFile(i) = 0;
        continue;
    end

    N = sB - sA + 1; % 裁剪后的长度（样本点）

    % 裁剪后不足一个窗口就跳过
    if N < winS
        winsPerFile(i) = 0;
    else
        winsPerFile(i) = floor((N - winS)/stepS) + 1;
    end
end
totalWins = sum(winsPerFile);

% 输出结构预分配
feat = struct();
feat.X = zeros(totalWins, D);
feat.featureNames = featureNames;
feat.fileIndex   = zeros(totalWins,1);
feat.windowIndex = zeros(totalWins,1);
feat.sampleStart = zeros(totalWins,1);
feat.sampleEnd   = zeros(totalWins,1);

% -------- 主循环：逐文件、逐窗口提特征 --------
row = 0;
for i = 1:nFiles
    x0 = out.allEmgFilt{i};
x0 = x0(:, opts.useChannels); % 原始整段 N0×C
N0 = size(x0,1);
C  = size(x0,2);

% 裁剪有效区间
sA = trimStartS + 1;
sB = N0 - trimEndS;
if sB <= sA
    continue; % 这个文件裁剪后无有效数据
end

x = x0(sA:sB, :); % 裁剪后的信号 N×C

K = winsPerFile(i);
for w = 1:K
    s0 = (w-1)*stepS + 1;
    s1 = s0 + winS - 1;

    seg = x(s0:s1, :);

    row = row + 1;
    feat.fileIndex(row) = i;
    feat.windowIndex(row) = w;

    % 这里记录“原始文件索引”，方便你后续对应回时间戳
    feat.sampleStart(row) = (sA - 1) + s0;
    feat.sampleEnd(row)   = (sA - 1) + s1;

        % 计算该窗口特征行
        featRow = zeros(1, D);
        col = 0;

        for c = 1:C
            xc = seg(:,c);

            % ----- 时域特征 -----
            for k = 1:numel(timeList)
                col = col + 1;
                featRow(col) = computeTimeFeature(xc, timeList{k}, opts.zcThreshold, opts.sscThreshold);
            end

            % ----- 频域特征 -----
            for k = 1:numel(freqList)
                col = col + 1;
                featRow(col) = computeFreqFeature(xc, freqList{k}, fs, opts);
            end
        end

        feat.X(row,:) = featRow;
    end
end

end


% ===================== 子函数：时域特征 =====================
function v = computeTimeFeature(x, name, zcThr, sscThr)
% x: 窗口信号列向量
switch upper(name)
    case 'MAV'   % Mean Absolute Value
        v = mean(abs(x));

    case 'RMS'   % Root Mean Square
        v = sqrt(mean(x.^2));

    case 'WL'    % Waveform Length
        v = sum(abs(diff(x)));

    case 'ZC'    % Zero Crossing（可加阈值抑制噪声）
        % 统计符号变化且幅值差超过阈值
        dx = diff(x);
        s  = x(1:end-1).*x(2:end) < 0; % 符号翻转
        amp = abs(dx) > zcThr;
        v = sum(s & amp);

    case 'SSC'   % Slope Sign Changes
        d1 = diff(x);
        s = (d1(1:end-1).*d1(2:end)) < 0;
        amp = (abs(d1(1:end-1)) > sscThr) | (abs(d1(2:end)) > sscThr);
        v = sum(s & amp);

    otherwise
        error('Unknown time feature: %s', name);
end
end


% ===================== 子函数：频域特征 =====================
function v = computeFreqFeature(x, name, fs, opts)
% 用 Welch/FFT 得到功率谱后计算频域特征
% 这里采用简单 FFT 周期图；你也可以改成 pwelch 更稳健

N = numel(x);

% nfft 默认取 >=N 的 2次幂
if isfield(opts,'nfft') && ~isempty(opts.nfft)
    nfft = opts.nfft;
else
    nfft = 2^nextpow2(N);
end

X = fft(x, nfft);
P2 = abs(X/nfft).^2;          % 功率谱（未做窗函数的简化版）
P1 = P2(1:nfft/2+1);
f  = fs*(0:(nfft/2))/nfft;

% 避免全0导致 NaN
P1 = P1 + eps;

switch upper(name)
    case 'MNF'  % Mean Frequency
        v = sum(f(:).*P1(:)) / sum(P1(:));

    case 'MDF'  % Median Frequency
        c = cumsum(P1(:));
        idx = find(c >= 0.5*c(end), 1, 'first');
        v = f(idx);

    case 'TTP'  % Total Power (Total Spectrum Power)
        v = sum(P1);

    otherwise
        error('Unknown freq feature: %s', name);
end
end