%% 跨位置验证对比实验
clear; clc;

fprintf('\n');
fprintf('========================================\n');
fprintf('    跨位置验证对比实验\n');
fprintf('========================================\n\n');

% ============================================================
% 第1步：加载之前的分析结果
% ============================================================
fprintf('加载数据...\n');

if ~exist('feature_robustness_report.mat', 'file')
    error('请先运行 run_robustness_analysis.m 生成分析报告���');
end

load('feature_robustness_report.mat', 'featureData', 'report');
fprintf('✓ 已加载特征数据和鲁棒性分析报告\n\n');

% ============================================================
% 第2步：方案A - 使用全部42个特征
% ============================================================
fprintf('========================================\n');
fprintf('方案A: 使用全部 42 个特征\n');
fprintf('========================================\n');

opts_all = struct();
opts_all.selectedFeatures = 1:42;  % 全部特征

results_all = runCrossPositionValidation(featureData, opts_all);

% ============================================================
% 第3步：方案B - 使用Top-20鲁棒特征
% ============================================================
fprintf('========================================\n');
fprintf('方案B: 使用Top-20 鲁棒特征\n');
fprintf('========================================\n');

opts_top20 = struct();
opts_top20.selectedFeatures = report.rankedFeatures(1:20);

results_top20 = runCrossPositionValidation(featureData, opts_top20);

% ============================================================
% 第4步：性能对比
% ============================================================
fprintf('\n');
fprintf('========================================\n');
fprintf('    性能对比总结\n');
fprintf('========================================\n\n');

acc_all = [results_all.accuracy];
acc_top20 = [results_top20.accuracy];

fprintf('方案A (全部42特征):\n');
fprintf('  平均准确率: %.2f%%\n', mean(acc_all)*100);
fprintf('  标准差:     %.2f%%\n', std(acc_all)*100);
fprintf('  最高/最低:  %.2f%% / %.2f%%\n\n', max(acc_all)*100, min(acc_all)*100);

fprintf('方案B (Top-20特征):\n');
fprintf('  平均准确率: %.2f%%\n', mean(acc_top20)*100);
fprintf('  标准差:     %.2f%%\n', std(acc_top20)*100);
fprintf('  最高/最低:  %.2f%% / %.2f%%\n\n', max(acc_top20)*100, min(acc_top20)*100);

improvement = (mean(acc_top20) - mean(acc_all)) * 100;
fprintf('性能变化: %.2f 个百分点\n', improvement);

if improvement > 0
    fprintf('✅ 筛选特征后性能提升！\n');
elseif improvement < -2
    fprintf('⚠️  筛选特征后性能下降较多\n');
else
    fprintf('➡️  筛选特征后性能基本持平\n');
end

fprintf('========================================\n\n');

% ============================================================
% 第5步：可视化对比
% ============================================================
figure('Position', [100 100 1200 500]);

% 子图1: 逐位置准确率对比
subplot(1,2,1);
positions = [results_all.testPosition];
plot(positions, acc_all*100, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '全部42特征');
hold on;
plot(positions, acc_top20*100, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Top-20特征');
hold off;
xlabel('测试位置');
ylabel('准确率 (%)');
title('跨位置准确率对比');
legend('Location', 'best');
grid on;
ylim([0 100]);

% 子图2: 箱线图对比
subplot(1,2,2);
boxplot([acc_all'*100, acc_top20'*100], ...
    'Labels', {'全部42特征', 'Top-20特征'});
ylabel('准确率 (%)');
title('准确率分布对比');
grid on;
ylim([0 100]);

% 添加均值标记
hold on;
plot(1, mean(acc_all)*100, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(2, mean(acc_top20)*100, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
legend('均值');
hold off;

% ============================================================
% 第6步：保存结果
% ============================================================
save('cross_position_results.mat', 'results_all', 'results_top20');
fprintf('✅ 结果已保存: cross_position_results.mat\n\n');

% ============================================================
% 第7步：详细混淆矩阵（可选）
% ============================================================
fprintf('是否显示详细混淆矩阵？(y/n): ');
choice = input('', 's');

if strcmpi(choice, 'y')
    fprintf('\n========== 方案B (Top-20) 各位置混淆矩阵 ==========\n\n');
    
    for i = 1:length(results_top20)
        fprintf('位置 %d (准确率: %.2f%%):\n', ...
            results_top20(i).testPosition, ...
            results_top20(i).accuracy * 100);
        disp(results_top20(i).confusionMat);
        fprintf('  (行=真实, 列=预测)\n\n');
    end
end

fprintf('========================================\n');
fprintf('跨位置验证完成！\n');
fprintf('========================================\n');