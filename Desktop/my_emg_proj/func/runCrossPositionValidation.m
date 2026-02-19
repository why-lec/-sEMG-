function results = runCrossPositionValidation(featureData, opts)
% runCrossPositionValidation - 跨位置验证
% 
% 策略：Leave-One-Position-Out (LOPO)
%   - 用8个位置训练，1个位置测试
%   - 循环9次，每个位置都当一次测试集
%
% 输入：
%   featureData.features   - N×D 特征矩阵
%   featureData.gestures   - N×1 手势标签
%   featureData.positions  - N×1 位置标签
%   opts.selectedFeatures  - 使用哪些特征（索引）
%
% 输出：
%   results(i).testPosition - 测试的位置
%   results(i).accuracy     - 准确率
%   results(i).confusionMat - 混淆矩阵

if nargin < 2
    opts = struct();
end

% 默认使用全部特征
if ~isfield(opts, 'selectedFeatures')
    opts.selectedFeatures = 1:size(featureData.features, 2);
end

% 提取数据
X = featureData.features(:, opts.selectedFeatures);
Y_gesture = featureData.gestures;
Y_position = featureData.positions;

positions = unique(Y_position);
n_positions = length(positions);

fprintf('\n========================================\n');
fprintf('  跨位置验证 (LOPO)\n');
fprintf('========================================\n');
fprintf('特征数: %d\n', length(opts.selectedFeatures));
fprintf('位置数: %d\n', n_positions);
fprintf('策略: 每次用%d个位置训练，1个位置测试\n', n_positions-1);
fprintf('----------------------------------------\n\n');

results = struct();

for i = 1:n_positions
    test_pos = positions(i);
    
    % 划分训练/测试集
    test_idx = (Y_position == test_pos);
    train_idx = ~test_idx;
    
    X_train = X(train_idx, :);
    Y_train = Y_gesture(train_idx);
    X_test = X(test_idx, :);
    Y_test = Y_gesture(test_idx);
    
    fprintf('测试位置 %d: 训练集%d样本, 测试集%d样本\n', ...
        test_pos, sum(train_idx), sum(test_idx));
    
    % Z-score标准化（基于训练集）
    [X_train_z, mu, sigma] = zscore(X_train);
    X_test_z = (X_test - mu) ./ (sigma + eps);  % 用训练集的均值和标准差
    
    % 训练SVM
    t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
    mdl = fitcecoc(X_train_z, Y_train, 'Learners', t, 'ClassNames', [1 2 3]);
    
    % 预测
    Y_pred = predict(mdl, X_test_z);
    
    % 评估
    acc = mean(Y_pred == Y_test);
    C = confusionmat(Y_test, Y_pred);
    
    % 保存结果
    results(i).testPosition = test_pos;
    results(i).trainSize = sum(train_idx);
    results(i).testSize = sum(test_idx);
    results(i).accuracy = acc;
    results(i).confusionMat = C;
    results(i).Y_true = Y_test;
    results(i).Y_pred = Y_pred;
    
    fprintf('  → 准确率: %.2f%%\n\n', acc*100);
end

% 计算平均准确率
avg_acc = mean([results.accuracy]);

fprintf('========================================\n');
fprintf('平均跨位置准确率: %.2f%%\n', avg_acc*100);
fprintf('最高: %.2f%% (位置%d)\n', ...
    max([results.accuracy])*100, ...
    results([results.accuracy] == max([results.accuracy])).testPosition);
fprintf('最低: %.2f%% (位置%d)\n', ...
    min([results.accuracy])*100, ...
    results([results.accuracy] == min([results.accuracy])).testPosition);
fprintf('========================================\n\n');

end