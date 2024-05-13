%%
addpath('~/CoSMoMVPA/mvpa/')
addpath('~/Repository/CommonFunctions/matplotlib/')

load('../derivatives/results/rsa_stats.mat','stats')

%%
f=figure(3);clf
f.Position(3:4) = [900 700];

% time time correlation
subplot(2,2,4)  
RP = stats.R.*stats.thresholded_cluster_map;
imagesc(stats.infant_timevec,stats.adult_timevec,RP',[0 .02])
c=colorbar();
colormap viridis
a=gca;
a.YDir = 'normal';
title('RSA correlation infant-adult')
xlabel('time infant (ms)')
ylabel('time adult (ms)')
xlim(minmax(stats.infant_timevec))
ylim(minmax(stats.infant_timevec))
colormap viridis
hold on
axis square
plot(a.XLim,a.YLim,'r')

% adult model correlation
co=tab10();h=[];
for p=1:3
    t = 4-p;
    a=subplot(2,2,2);hold on
    mu = stats.r_adult(:,t);
    h(p) = plot(stats.adult_timevec,mu,'LineWidth',2,'Color',co(t,:));
    plot(stats.adult_timevec,0*stats.adult_timevec,'k--')
    idx = stats.p_adult(:,t)<.05;
    plot(stats.adult_timevec(idx),0*stats.adult_timevec(idx)-t*.008-.02,'.','Color',co(t,:),'MarkerSize',20)
    xlabel('time adult (ms)')
    ylabel('correlation')
    title('RSA model fit (adults)')
    xlim(minmax(stats.adult_timevec))
end
legend(h,fliplr(stats.modelnames))

% infant model correlation
co=tab10();
for t=1:3
    pp=[5 3 1];
    a=subplot(3,2,pp(t));hold on
    mu = stats.r_infant(:,t);
    plot(stats.infant_timevec,mu,'LineWidth',2,'Color',co(t,:))
    plot(stats.infant_timevec,0*stats.infant_timevec,'k--')
    idx = stats.p_infant(:,t)<.05;
    plot(stats.infant_timevec(idx),0*stats.infant_timevec(idx)-.015,'.','Color',co(t,:),'MarkerSize',20)
    legend(stats.modelnames{t},'Location','NW')
    xlabel('time infant (ms)')
    ylabel('correlation')
    title(sprintf('RSA model fit: %s (infants)',stats.modelnames{t}))
    xlim(minmax(stats.infant_timevec))
    ylim([-.02 .025])
end

%%
fn = './figures/figure3';
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=2;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
