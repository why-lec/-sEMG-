%{
clear; clc;

addpath(fullfile('C:\Users\18441\Desktop\my_emg_proj', "func"));

folder = 'C:\Users\18441\Desktop\抓\zhua_csv';

opts = struct();
opts.fs = 1926;
opts.timeCols = [1 3 5 7 9 11];
opts.emgCols  = [2 4 6 8 10 12];

% 带通参数(巴特沃斯滤波器)
opts.bandpass = [20 450];
opts.bpOrder  = 4;

% 陷波判据与参数
opts.enableNotchAuto = true; % 自动判定是否陷波
opts.notchFreq = 50;
opts.notchQ    = 35;
opts.notchThreshold_dB = 8;     % 8dB 是一个常用起点
opts.notchMajorityChannels = 3; % >=3个通道超过阈值则对该文件陷波

out = preprocessDelsysFolder(folder, opts);

% out 里会包含：
% out.files, out.allTime, out.allEmgRaw, out.allEmgBP, out.allEmgFilt
disp("Done.");

%特征处理
optsFeat = struct();
optsFeat.fs = 1926;        % 采样率，必须跟预处理一致
optsFeat.win_ms = 200;     % 窗口长度 200ms（常用）
optsFeat.step_ms = 50;     % 步长 50ms（常用）
optsFeat.trimStart_s = 1.5; % 去掉开头1.5秒
optsFeat.trimEnd_s   = 1.5; % 去掉结尾1.5秒

feat = extractEmgFeaturesFromOut(out, optsFeat);

%disp(size(feat.X));
%disp(feat.featureNames(1:min(10,numel(feat.featureNames))));

trialMean = summarizeByTrial(feat, numel(out.files), 'mean');
%disp(size(trialMean.X));   % 预期是 90×42
%disp(trialMean.X(1,1:5))

%绘制箱线图
plot_feature = 'RMS';%想要绘制的特征名
plotTrialFeatureBox(trialMean, plot_feature,1:6);

%}

clear; clc;

% 1) 把 func 文件夹加入路径，MATLAB 才能找到你写的函数
addpath(fullfile('C:\Users\18441\Desktop\my_emg_proj', "func"));

% 2) 三个手势的文件夹路径（你给过的3个路径）
folder_nie  = 'C:\Users\18441\Desktop\捏\nie_csv';
folder_zhua = 'C:\Users\18441\Desktop\抓\zhua_csv';
folder_wo   = 'C:\Users\18441\Desktop\握\wo_csv';

% ============================================================
% A) 预处理参数 optsPre（原来的 opts 改名为 optsPre，更清晰）
% ============================================================
optsPre = struct();              % 创建一个空结构体，用来装参数
optsPre.fs = 1926;               % 采样率
optsPre.timeCols = [1 3 5 7 9 11]; % 6个时间列（每个传感器一个）
optsPre.emgCols  = [2 4 6 8 10 12];% 6个EMG列（每个传感器一个）

% 带通参数(巴特沃斯滤波器)
optsPre.bandpass = [20 450];     % 带通范围20~450Hz（EMG常用）
optsPre.bpOrder  = 4;            % 4阶巴特沃斯

% 陷波判据与参数（自动判断是否需要50Hz陷波）
optsPre.enableNotchAuto = true;  % true=启用自动判定
optsPre.notchFreq = 50;          % 50Hz
optsPre.notchQ    = 35;          % 陷波Q值，越大越窄
optsPre.notchThreshold_dB = 8;   % 判据阈值（越小越容易触发陷波）
optsPre.notchMajorityChannels = 3; % >=3通道超过阈值才陷波

% ============================================================
% B) 特征参数 optsFeat（裁剪+滑窗+时域/频域特征）
% ============================================================
optsFeat = struct();         % 创建特征参数结构体
optsFeat.fs = 1926;          % 采样率要一致
optsFeat.win_ms = 200;       % 窗口长度（ms）
optsFeat.step_ms = 50;       % 步长（ms）

% 去掉静息段：前后1.5秒（你确认过要这样）
optsFeat.trimStart_s = 1.5;
optsFeat.trimEnd_s   = 1.5;

% 这里把特征列表写出来，保证你清楚自己在算什么
optsFeat.features = struct();
optsFeat.features.time = {'MAV','RMS','WL','ZC','SSC'}; % 5个时域
optsFeat.features.freq = {'MNF','MDF'};                 % 2个频域

% ============================================================
% C) 分别对三个手势提取 “trial级特征矩阵”
%    输出 X_* 形状大约是 90×42
% ============================================================
[X_nie, featureNames] = buildGestureTrialSet(folder_nie,  optsPre, optsFeat);
[X_zhua, ~]           = buildGestureTrialSet(folder_zhua, optsPre, optsFeat);
[X_wo, ~]             = buildGestureTrialSet(folder_wo,   optsPre, optsFeat);

% ============================================================
% D) 拼成一个总数据集（用于分类）
%    Xall: (总trial数)×42
%    Yall: (总trial数)×1，类别标签 1/2/3
% ============================================================
Xall = [X_nie; X_zhua; X_wo];

Yall = [ ...
    1 * ones(size(X_nie,1), 1); ...
    2 * ones(size(X_zhua,1),1); ...
    3 * ones(size(X_wo,1),  1) ...
];

% ============================================================
% E) 特征标准化（非常重要：不同特征量纲差异很大）
%    zscore: 每一列都变成 “均值0、方差1”
% ============================================================
[Xall_z, mu, sigma] = zscore(Xall); %#ok<NASGU>

% ============================================================
% F) 训练三分类 SVM（fitcecoc = 多分类框架，内部用多个二分类SVM）
% ============================================================
t = templateSVM( ...
    'KernelFunction', 'rbf', ... % RBF核（常用且效果一般不错）
    'KernelScale', 'auto', ...   % 自动核尺度
    'Standardize', false);       % 我们已经手动zscore了，所以这里false

Mdl = fitcecoc(Xall_z, Yall, ...
    'Learners', t, ...
    'ClassNames', [1 2 3]);

% ============================================================
% G) 用5折交叉验证评估（先做一个基线准确率）
% ============================================================
CV = crossval(Mdl, 'KFold', 5);  % ��数据分成5份轮流验证
yhat = kfoldPredict(CV);         % 得到每条样本的预测类别

acc = mean(yhat == Yall);        % 计算总体准确率
fprintf('5-fold CV accuracy = %.2f%%\n', 100*acc);

% 混淆矩阵：看每一类被错分到哪里
C = confusionmat(Yall, yhat);
disp('Confusion matrix (rows=true, cols=pred):');
disp(C);

% 画混淆矩阵图（可视化更直观）
figure;
confusionchart(Yall, yhat, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title('SVM (ECOC) 5-fold CV confusion chart');

% ============================================================
% H) （可选）你之前的箱线图还想画也行：
%     例如画 “捏” 这个手势的 RMS 6通道箱线图
% ============================================================
% trialMeanNie = struct('X', X_nie, 'featureNames', featureNames);
% plotTrialFeatureBox(trialMeanNie, 'RMS', 1:6);