function [Xtrial, featureNames] = buildGestureTrialSet(folder, optsPre, optsFeat)
% buildGestureTrialSet
% ============================================================
% 输入一个手势的 csv 文件夹，完成：
%   1) 预处理（带通 + 可选自动陷波）
%   2) 裁剪静息段（你现在是前后1.5s）
%   3) 切窗提特征（时域+频域）
%   4) 对每个 trial 的所有窗口特征求均值，得到 trial 级特征
%
% 输出
%   Xtrial       : nTrials×D（例如 90×42）
%   featureNames : 1×D
% ============================================================

% 1) 预处理：读入并滤波，输出 out（包含 out.allEmgFilt）
out = preprocessDelsysFolder(folder, optsPre);

% 2) 提取窗口级特征：输出 feat.X (总窗口数×D)，并带 fileIndex
feat = extractEmgFeaturesFromOut(out, optsFeat);

% 3) trial级汇总：对每个文件（trial）内的窗口特征做均值
trialMean = summarizeByTrial(feat, numel(out.files), 'mean');

Xtrial = trialMean.X;
featureNames = trialMean.featureNames;

end