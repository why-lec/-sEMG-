function trial = summarizeByTrial(feat, nFiles, method)
% summarizeByTrial
% ============================================================
% 把“窗口级特征”汇总成“trial级特征”。
%
% 为什么要做：
%   - 一个 trial 会产生很多窗口（你现在总共13230个窗口）
%   - 但你想比较的是 90 个 trial 之间每个特征的统计差异
%   - 所以对每个 trial 内的窗口特征做均值/标准差等汇总
%
% 输入
%   feat   : extractEmgFeaturesFromOut 输出的结构体
%            feat.X 是 (总窗口数)×D
%            feat.fileIndex 指每行来自哪个文件（1..nFiles）
%   nFiles : trial 数量（例如 90）
%   method : 'mean' | 'std' | 'median'
%
% 输出
%   trial : 结构体
%     trial.X : nFiles×D，每个 trial 一行
%     trial.featureNames : 与 feat.featureNames 相同
% ============================================================

if nargin < 3
    method = 'mean'; % 如果你不传method，默认取均值
end

X = feat.X;                 % 窗口级特征矩阵
idx = feat.fileIndex;       % 每个窗口属于哪个 trial
D = size(X,2);              % 特征维数（你这里是42）

% 先用 NaN 预填充，避免某个trial没有窗口时出错
Xtrial = nan(nFiles, D);

for i = 1:nFiles
    Xi = X(idx == i, :);    % 取出第 i 个trial 的所有窗口特征（Ki×D）
    if isempty(Xi)
        continue;           % 如果该trial没有窗口，保持NaN
    end

    switch lower(method)
        case 'mean'
            Xtrial(i,:) = mean(Xi, 1, 'omitnan');   % 对每一列取均值
        case 'std'
            Xtrial(i,:) = std(Xi, 0, 1, 'omitnan'); % 对每一列取标准差
        case 'median'
            Xtrial(i,:) = median(Xi, 1, 'omitnan'); % 对每一列取中位数
        otherwise
            error('Unknown method: %s', method);
    end
end

trial = struct();
trial.X = Xtrial;
trial.featureNames = feat.featureNames;

end