function [needNotch, Rdb_ch] = needNotch50Hz(x_bp, fs, notchFreq, threshold_dB, majorityChannels)
% needNotch50Hz
% 输入带通后的EMG (N x C)，输出每通道 50Hz 峰值相对背景能量比（dB），
% 并用“多数通道超过阈值”规则判断是否需要陷波。

bandTarget = [notchFreq-1 notchFreq+1];
bandBg1    = [notchFreq-5 notchFreq-1];
bandBg2    = [notchFreq+1 notchFreq+5];

nfft = 4096;
win  = 2048;
noverlap = 1024;

C = size(x_bp,2);
Rdb_ch = nan(1,C);

for ch = 1:C
    [Pxx,f] = pwelch(x_bp(:,ch), win, noverlap, nfft, fs);

    PT  = bandpower(Pxx, f, bandTarget, 'psd');
    PBg = (bandpower(Pxx, f, bandBg1, 'psd') + bandpower(Pxx, f, bandBg2, 'psd'))/2;

    Rdb_ch(ch) = 10*log10(PT) - 10*log10(PBg);
end

needNotch = sum(Rdb_ch > threshold_dB) >= majorityChannels;

end