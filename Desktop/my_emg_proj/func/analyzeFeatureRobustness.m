function report = analyzeFeatureRobustness(featureData, opts)
% analyzeFeatureRobustness - 分析特征的鲁棒性
%
% 输入：
%   featureData.features   - N×42 特征矩阵
%   featureData.gestures   - N×1 手势标签 (1/2/3)
%   featureData.positions  - N×1 位置标签 (1~9)
%   featureData.featureNames - 特征名称
%
% 输出：
%   report.fisherScores - Fisher判别比（类间可分性）
%   report.cvScores - 变异系数（位置稳定性）
%   report.robustnessScores - 综合评分
%   report.rankedFeatures - 按评分排序的特征索引

% 默认参数
if nargin < 2 %nargin代表传入函数参数个数
    opts = struct(); %创建一个空结构体opts
end
if ~isfield(opts, 'fisherWeight'), opts.fisherWeight = 0.6; end %isfield()检查结构体中是否含有该字段
if ~isfield(opts, 'cvWeight'), opts.cvWeight = 0.4; end

% 提取数据
X = featureData.features;
Y_gesture = featureData.gestures;
Y_position = featureData.positions;
featureNames = featureData.featureNames;

[N, D] = size(X);

fprintf('\n========================================\n');
fprintf('  特征鲁棒性分析\n');
fprintf('========================================\n');
fprintf('数据量: %d trials\n', N);
fprintf('特征数: %d\n', D);
fprintf('手势: %d类\n', length(unique(Y_gesture)));
fprintf('位置: %d个\n', length(unique(Y_position)));
fprintf('----------------------------------------\n');

% 计算每个特征的评分
fisherScores = zeros(1, D);
cvScores = zeros(1, D);

fprintf('正在计算特征评分...\n');

for f = 1:D
    feature_vals = X(:, f);
    
    % Fisher判别比
    fisherScores(f) = calculateFisherScore(feature_vals, Y_gesture);
    
    % 变异系数
    cvScores(f) = calculatePositionCV(feature_vals, Y_gesture, Y_position);
    
    if mod(f, 10) == 0
        fprintf('  进度: %d/%d\n', f, D);
    end
end

fprintf('计算完成！\n');

% 归一化并综合评分
fisher_norm = fisherScores / (max(fisherScores) + eps);
cv_norm = 1 ./ (1 + cvScores);
robustnessScores = opts.fisherWeight * fisher_norm + opts.cvWeight * cv_norm;

% 排序
[~, rankedIdx] = sort(robustnessScores, 'descend');

% 输出结果
report = struct();
report.fisherScores = fisherScores;
report.cvScores = cvScores;
report.robustnessScores = robustnessScores;
report.rankedFeatures = rankedIdx;
report.featureNames = featureNames;

% 打印Top-20
fprintf('\n========== Top-20 鲁棒特征 ==========\n');
fprintf('排名\t特征名\t\t\tFisher\tCV\t综合分\n');
fprintf('================================================\n');

for i = 1:min(20, D)
    idx = rankedIdx(i);
    fprintf('%2d\t%-20s\t%.3f\t%.3f\t%.3f\n', ...
        i, featureNames{idx}, ...
        fisherScores(idx), cvScores(idx), robustnessScores(idx));
end

fprintf('================================================\n\n');

end

% ========== 子函数1: Fisher判别比 ==========
function fisher = calculateFisherScore(vals, labels)
classes = unique(labels);
mu_total = mean(vals);

% 类间方差
sb = 0;
for k = 1:length(classes)
    idx = (labels == classes(k));
    mu_k = mean(vals(idx));
    n_k = sum(idx);
    sb = sb + n_k * (mu_k - mu_total)^2;
end
sb = sb / length(vals);

% 类内方差
sw = 0;
for k = 1:length(classes)
    idx = (labels == classes(k));
    sw = sw + var(vals(idx)) * sum(idx);
end
sw = sw / length(vals);

fisher = sb / (sw + eps);
end

% ========== 子函数2: 变异系数 ==========
function cv_avg = calculatePositionCV(vals, gestures, positions)
unique_gestures = unique(gestures);
cv_list = [];

for g = unique_gestures'
    idx = (gestures == g);
    gesture_vals = vals(idx);
    
    mu = mean(gesture_vals);
    sigma = std(gesture_vals);
    cv = sigma / (abs(mu) + eps);
    
    cv_list = [cv_list; cv]; %#ok<AGROW>
end

cv_avg = mean(cv_list);
end