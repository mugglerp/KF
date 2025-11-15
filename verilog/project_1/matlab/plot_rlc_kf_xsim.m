%% RLC KF plotter (run this in the xsim run folder)
% Auto-detect latest CSV/TXT with expected headers, then plot 4 panels.

function plot_rlc_kf_xsim()
clc; close all;

% -------- locate latest file in current (xsim) folder --------
files = [dir('*.csv'); dir('*.txt')];
assert(~isempty(files), '未找到数据文件(.csv/.txt)。把 MATLAB 当前目录切到 xsim 数据所在处。');
[~, idx] = max([files.datenum]);
fname = files(idx).name;

% -------- read table robustly --------
opts = detectImportOptions(fname, 'NumHeaderLines', 0);
T = readtable(fname, opts);

% 必须列：k, il, il_hat, vc, vc_hat, vR2, vR2_hat, iR2, iR2_hat
need = ["k","il","il_hat","vc","vc_hat","vR2","vR2_hat","iR2","iR2_hat"];
have = string(T.Properties.VariableNames);
missing = setdiff(need, have);
assert(isempty(missing), "文件缺少列: " + strjoin(missing, ", "));

k        = T.k;
il       = T.il;       il_hat   = T.il_hat;
vc       = T.vc;       vc_hat   = T.vc_hat;
vR2      = T.vR2;      vR2_hat  = T.vR2_hat;
iR2      = T.iR2;      iR2_hat  = T.iR2_hat;

% 可选：只看前 100 个采样（复现论文图的窗口）
Nwin = min(100, height(T));
kk   = k(1:Nwin);

% -------- make 4 panels,黑=真值, 蓝=估计 --------
figure('Color','w','Position',[100 100 1200 720]);

% (a) il
subplot(2,2,1);
plot(kk, il(1:Nwin), 'k','LineWidth',1.5); hold on;
plot(kk, il_hat(1:Nwin), 'b','LineWidth',1.5);
grid on; xlabel('Samples'); ylabel('Current (A)');
legend('Inductor current (i_l)','Estimated current','Location','northwest');
text(kk(3), min(ylim)+0.1*range(ylim), 'a)','FontSize',14,'FontWeight','bold');

% (b) vc
subplot(2,2,2);
plot(kk, vc(1:Nwin), 'k','LineWidth',1.5); hold on;
plot(kk, vc_hat(1:Nwin), 'b','LineWidth',1.5);
grid on; xlabel('Samples'); ylabel('Voltage (V)');
legend('Capacitor voltage (v_c)','Estimated voltage','Location','northwest');
text(kk(3), min(ylim)+0.1*range(ylim), 'b)','FontSize',14,'FontWeight','bold');

% (c) vR2
subplot(2,2,3);
plot(kk, vR2(1:Nwin), 'k','LineWidth',1.5); hold on;
plot(kk, vR2_hat(1:Nwin), 'b','LineWidth',1.5);
grid on; xlabel('Samples'); ylabel('Voltage (V)');
legend('Resistor voltage (v_{R2})','Estimated voltage','Location','northwest');
text(kk(3), min(ylim)+0.1*range(ylim), 'c)','FontSize',14,'FontWeight','bold');

% (d) iR2
subplot(2,2,4);
plot(kk, iR2(1:Nwin), 'k','LineWidth',1.5); hold on;
plot(kk, iR2_hat(1:Nwin), 'b','LineWidth',1.5);
grid on; xlabel('Samples'); ylabel('Current (A)');
legend('Resistor current (i_{R2})','Estimated current','Location','northwest');
text(kk(3), min(ylim)+0.1*range(ylim), 'd)','FontSize',14,'FontWeight','bold');

sgtitle(sprintf('RLC KF — %s (showing first %d samples)', fname, Nwin), 'FontWeight','bold');

% -------- 误差指标（可选打印）--------
rms_il  = rms(il_hat - il);
rms_vc  = rms(vc_hat - vc);
rms_vR2 = rms(vR2_hat - vR2);
rms_iR2 = rms(iR2_hat - iR2);
fprintf('RMS errors: il=%.6g, vc=%.6g, vR2=%.6g, iR2=%.6g\n', ...
        rms_il, rms_vc, rms_vR2, rms_iR2);
end

% 直接运行：
% >> plot_rlc_kf_xsim
