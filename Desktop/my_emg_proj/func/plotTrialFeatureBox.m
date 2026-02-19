function plotTrialFeatureBox(trialFeat, baseFeatureName, channels)
% plotTrialFeatureBox
% ============================================================
% 对 trial 级特征矩阵中的“同一特征(如RMS)的多个通道(ch1~ch6)”，
% 画在同一张图里的箱线图 + 散点。
%
% 举例：
%   baseFeatureName = 'RMS'
% 会自动寻找：
%   'ch1_RMS','ch2_RMS',...,'ch6_RMS'
%
% 输入
%   trialFeat        : summarizeByTrial 输出结构体
%   baseFeatureName  : 特征基名，例如 'RMS' / 'MAV' / 'WL' / 'ZC' / 'SSC'
%   channels         : 要画哪些通道，例如 1:6（可选，默认 1:6）
% ============================================================

if nargin < 3 || isempty(channels)
    channels = 1:6; % 默认画 ch1~ch6
end

% ---- 1) 构造要找的特征全名列表：chX_baseFeatureName ----
nCh = numel(channels);
fullNames = cell(1, nCh);
for k = 1:nCh
    fullNames{k} = sprintf('ch%d_%s', channels(k), baseFeatureName);
end

% ---- 2) 找到每个 fullName 在 featureNames 中对应的列号 ----
idx = nan(1, nCh);
for k = 1:nCh
    idx(k) = find(strcmp(trialFeat.featureNames, fullNames{k}), 1);
    if isnan(idx(k))
        error('Feature "%s" not found. Check spelling and featureNames.', fullNames{k});
    end
end

% ---- 3) 取出数据：nTrials×nCh，每列一个通道 ----
V = trialFeat.X(:, idx);

% ---- 4) 画箱线图：一次画多列（每列一组箱线）----
figure;
boxplot(V, 'Labels', fullNames); % labels显示 ch1_RMS...ch6_RMS
title(['Trial summary (channels): ' baseFeatureName]);
ylabel('Value');
grid on;

% ---- 5) 叠加散点：每个trial在每个通道上的点都画出来 ----
hold on;
nTrials = size(V,1);
for k = 1:nCh
    x = k * ones(nTrials,1);      % 第k组的x位置
    y = V(:,k);                   % 第k组的数据
    plot(x, y, 'k.', 'MarkerSize', 8);
end
hold off;

end