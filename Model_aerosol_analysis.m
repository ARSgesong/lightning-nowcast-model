clear all
applyset = h5read('GLM_applyset_CAMS_20230515.h5','/applyset');
applyset = applyset';
applyset2 = h5read('GLM_applyset_CAMS_noAOD_20230515.h5','/applyset');
applyset2  =applyset2';

applyset = applyset(applyset(:,1)>0,:);
applyset = sortrows(applyset,[1 2 3]);
applyset2 = applyset2(applyset2(:,1)>0,:);
applyset2 = sortrows(applyset2,[1 2 3]);

applyset(:,17) = applyset(:,16)>0.45;
applyset2(:,11) = applyset2(:,10)>0.45;

set(0,'DefaultAxesFontname','Arial')
set(0,'DefaultTextFontname','Arial')

applyset(:,19) = applyset(:,13) + applyset(:,9) + applyset(:,10) + applyset(:,11) + applyset(:,12);
applyset2 = applyset2(applyset(:,19)<1,:);
applyset = applyset(applyset(:,19)<1,:);
result = applyset;

testset = result(:,15);
predset = result(:,17);

testset2 = applyset2(:,9);
predset2 = applyset2(:,11);

aerosol  = result(:,19);

nbins = 50;

t = tiledlayout(2,1);
ax1=subplot(2,1,1,'position',[0.1  0.65  0.8  0.3]);
hold on
h = histogram(aerosol,nbins);
h.FaceColor = [0.1200    0.5600    0.5500];
% h.EdgeColor = 'b';
grid on
grid minor
xlim([0 1]);
set(ax1,'FontName','Arial','FontWeight','bold','FontSize',10)
set(gca,'XTickLabel',' ')
box off
axx1 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
set(axx1,'YTick', []);
set(axx1,'XTick', []);
box on

[N,edges,bin] = histcounts(aerosol,nbins);
for i = 1:nbins
    bin_in = find(bin(:) == i);
    bin_test = testset(bin_in);
    bin_pred = predset(bin_in);
    POD = mean((bin_test - bin_pred));
    TP = bin_test(bin_test == 1 & bin_pred == 1,:);
    FN = bin_test(bin_test == 1 & bin_pred == 0,:);
    FP = bin_test(bin_test == 0 & bin_pred == 1,:);
    POD = length(TP)/(length(TP) + length(FN));
    FAR = 1 - length(TP)/(length(TP) + length(FP));
    CSI = length(TP)/(length(TP) + length(FN)+ length(FP));
    record(i,1) = mean([edges(i) edges(i+1)]);
    record(i,2) = POD;
    record(i,3) = FAR;
    record(i,4) = CSI;
    record(i,5) = length(bin_in);

    bin_test2 = testset2(bin_in);
    bin_pred2 = predset2(bin_in);
    POD = mean((bin_test2 - bin_pred2));
    TP = bin_test2(bin_test2 == 1 & bin_pred2 == 1,:);
    FN = bin_test2(bin_test2 == 1 & bin_pred2 == 0,:);
    FP = bin_test2(bin_test2 == 0 & bin_pred2 == 1,:);
    POD = length(TP)/(length(TP) + length(FN));
    FAR = 1 - length(TP)/(length(TP) + length(FP));
    CSI = length(TP)/(length(TP) + length(FN)+ length(FP));
    record(i,7) = mean([edges(i) edges(i+1)]);
    record(i,8) = POD;
    record(i,9) = FAR;
    record(i,10) = CSI;
    record(i,11) = length(bin_in);
end

ax2=subplot(2,1,2,'position',[0.1  0.1  0.8  0.5]);
hold on
record = record(~isnan(record(:,2)),:);
yyaxis left
plot(record(:,1),record(:,4),'-','linewidth',1.5);
plot(record(:,1),record(:,10),'--','linewidth',1.5);
ylabel('CSI')
yyaxis right
plot(record(:,1),record(:,3),'-','linewidth',1.5);
plot(record(:,1),record(:,9),'--','linewidth',1.5);
ylabel('FAR')
grid on
grid minor
xlim([0 1]);
% Link the axes
linkaxes([ax1,ax2],'x');
% Add shared title and axis labels
xlabel('AOD')
set(ax2,'FontName','Arial','FontWeight','bold','FontSize',10)
% legend('CSI','FAR','Location','northwest','FontSize',12)
box off
axx2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
set(axx2,'YTick', []);
set(axx2,'XTick', []);
box on

set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02))


% % Move plots closer together
% xticklabels(ax1,{})
% t.TileSpacing = 'compact';
