%% 多位置特征鲁棒性分析
clear; clc;

addpath(fullfile('C:\Users\18441\Desktop\my_emg_proj', "func"));

fprintf('========================================\n');
fprintf('多位置特征鲁棒性分析\n');
fprintf('========================================\n\n');

% ============================================================
% 第1步：定义数据路径
% ============================================================
folder_nie  = 'C:\Users\18441\Desktop\捏\nie_csv';
folder_zhua = 'C:\Users\18441\Desktop\抓\zhua_csv';
folder_wo   = 'C:\Users\18441\Desktop\握\wo_csv';

% ============================================================
% 第2步：设置参数（与之前相同）
% ============================================================
optsPre = struct();
optsPre.fs = 1926;
optsPre.timeCols = [1 3 5 7 9 11];
optsPre.emgCols  = [2 4 6 8 10 12];
optsPre.bandpass = [20 450];
optsPre.bpOrder  = 4;
optsPre.enableNotchAuto = true;
optsPre.notchFreq = 50;
optsPre.notchQ    = 35;
optsPre.notchThreshold_dB = 8;
optsPre.notchMajorityChannels = 3;

optsFeat = struct();
optsFeat.fs = 1926;
optsFeat.win_ms = 200;
optsFeat.step_ms = 50;
optsFeat.trimStart_s = 1.5;
optsFeat.trimEnd_s   = 1.5;
optsFeat.features = struct();
optsFeat.features.time = {'MAV','RMS','WL','ZC','SSC'};
optsFeat.features.freq = {'MNF','MDF'};

% ============================================================
% 第3步：提取特征
% ============================================================
fprintf('步骤1: 提取特征...\n');
fprintf('----------------------------------------\n');

[X_nie, featureNames] = buildGestureTrialSet(folder_nie, optsPre, optsFeat);
fprintf('✓ 捏: %d trials\n', size(X_nie, 1));

[X_zhua, ~] = buildGestureTrialSet(folder_zhua, optsPre, optsFeat);
fprintf('✓ 抓: %d trials\n', size(X_zhua, 1));

[X_wo, ~] = buildGestureTrialSet(folder_wo, optsPre, optsFeat);
fprintf('✓ 握: %d trials\n\n', size(X_wo, 1));

% ============================================================
% 第4步：生成位置标签（每个位置10个trial）
% ============================================================
fprintf('步骤2: 生成位置标签...\n');
fprintf('----------------------------------------\n');

trials_per_position = 10;

% 自动生成：trial 1-10→位置1, 11-20→位置2, ...
position_labels_nie  = ceil((1:size(X_nie,1))  / trials_per_position)';
position_labels_zhua = ceil((1:size(X_zhua,1)) / trials_per_position)';
position_labels_wo   = ceil((1:size(X_wo,1))   / trials_per_position)';

fprintf('✓ 捏的位置: %d ~ %d\n', min(position_labels_nie), max(position_labels_nie));
fprintf('✓ 抓的位置: %d ~ %d\n', min(position_labels_zhua), max(position_labels_zhua));
fprintf('✓ 握的位置: %d ~ %d\n\n', min(position_labels_wo), max(position_labels_wo));

% ============================================================
% 第5步：合并数据
% ============================================================
fprintf('步骤3: 合并数据...\n');
fprintf('----------------------------------------\n');

Xall = [X_nie; X_zhua; X_wo];

Yall_gesture = [
    1 * ones(size(X_nie,1), 1);
    2 * ones(size(X_zhua,1), 1);
    3 * ones(size(X_wo,1), 1)
];

Yall_position = [
    position_labels_nie;
    position_labels_zhua;
    position_labels_wo
];

fprintf('✓ 总数据: %d trials × %d features\n', size(Xall, 1), size(Xall, 2));
fprintf('✓ 手势分布: 捏=%d, 抓=%d, 握=%d\n', ...
    sum(Yall_gesture==1), sum(Yall_gesture==2), sum(Yall_gesture==3));
fprintf('✓ 位置数量: %d 个\n\n', length(unique(Yall_position)));

% ============================================================
% 第6步：调用分析函数
% ============================================================
fprintf('步骤4: 开始鲁棒性分析...\n');
fprintf('----------------------------------------\n');

featureData = struct();
featureData.features = Xall;
featureData.gestures = Yall_gesture;
featureData.positions = Yall_position;
featureData.featureNames = featureNames;

report = analyzeFeatureRobustness(featureData);

% ============================================================
% 第7步：保存结果
% ============================================================
save('feature_robustness_report.mat', 'report', 'featureData');
fprintf('\n✅ 完整报告已保存: feature_robustness_report.mat\n');

% 保存Top-20特征索引
topK = 20;
selected_features = report.rankedFeatures(1:topK);
save('selected_robust_features.mat', 'selected_features', 'featureNames');
fprintf('✅ 筛选特征已保存: selected_robust_features.mat\n');

fprintf('\n========================================\n');
fprintf('分析完成！\n');
fprintf('========================================\n');