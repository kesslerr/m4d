
%%
addpath('~/CoSMoMVPA/mvpa')
addpath('~/fieldtrip');ft_defaults;
addpath('~/Repository/CommonFunctions/matplotlib/')

%%
datapath = '../';
P = readtable(sprintf('%s/participants.tsv',datapath),'FileType','text','Delimiter','\t');
X=[];
subinfo={};
ft_cell={};
subjectlist=[];
for subjectnr = 1:51
    fprintf('processing sub-%03i\n',subjectnr);tic;
    %% get files
    cosmofn = sprintf('%s/derivatives/cosmomvpa/sub-%03i_cosmomvpa_clean_rawdata.mat',datapath,subjectnr);
    if exist(cosmofn,'file')
        %% get fft
        x=load(cosmofn);
        ds = cosmo_meeg_baseline_correct(x.ds,[-.1 0],'absolute');
        dst = cosmo_dim_transpose(ds,'chan');
        dst.samples(any(abs(dst.samples)>50,2),:)=0;
        ds = cosmo_dim_transpose(dst,'chan');
        ds = cosmo_fx(ds,@mean);
        ds = cosmo_dim_transpose(ds,'time');
        ds = cosmo_dim_transpose(ds,'time');
        ft=cosmo_map2meeg(cosmo_dim_prune(cosmo_slice(ds,ds.fa.time>50,2)));
        ft.time = ft.time./1000;
        cfg=[];
        cfg.method = 'mtmfft';
        cfg.taper = 'hanning';
        out = ft_freqanalysis(cfg, ft);
        X(end+1,:) = mean(out.powspctrm);
        ft_cell{end+1} = out;
        subjectlist(end+1) = subjectnr;
        subinfo{end+1} = table2struct(P(strcmp(P.participant_id,sprintf('sub-%.03i',subjectnr)),:));
    end
end

%%
XX=X;
pidx = ~mod(ft_cell{1}.freq,5)& ft_cell{1}.freq>0 & ft_cell{1}.freq<=50;
layout = ft_prepare_layout(struct('layout','eeg1010.lay'));
f=figure(2);clf
f.Position(3:4) = [750 900];
p=0;co=tab10();
for s=1:numel(subjectlist) % 42 plots
    p = p+1;
    nrows = 8;
    ncols = 6;
    while ismember(p,[1:3 7:9]) %skip to insert average later
        p=p+1;
    end
    row = ceil(p/ncols);
    col = mod(p-1,ncols);
    a = axes('Position',[.01+col*1/ncols,1-row*.95/nrows .9/ncols .75/nrows]);
    stem(ft_cell{s}.freq,XX(s,:),'filled','Color',co(1,:),'LineWidth',1);hold on
    stem(ft_cell{s}.freq(pidx),XX(s,pidx),'filled','Color',co(4,:),'LineWidth',1);
    xlim([1 40])
    xticks(5:5:35);
    if row<nrows
        a.XTickLabel=[];
    else
        if col==2
            xlabel('Frequency (Hz)')
        end
    end
    tx = title(sprintf('%s (%i months)',subinfo{s}.participant_id,round(subinfo{s}.age_months)));
    a.YTick=[];
    if 1 % topomap
        a2=axes('Position',[a.Position(1:2) .6*a.Position(3) .6*a.Position(4)]+[.4*a.Position(3) .35*a.Position(4) 0 0]);
        cfg=[];
        cfg.xlim = [5 5];
        cfg.layout = layout;
        cfg.figure = a2;
        cfg.comment='no';
        cfg.style = 'straight';
        cfg.colormap = [linspace(1,0.8392,100);...
            linspace(1,0.1529,100);...
            linspace(1,0.1569,100)]';
        ft_topoplotER(cfg,ft_cell{s});
    end
end

% average
a=axes('Position',[.025 .8  3*.9/ncols .175]);
stem(ft_cell{1}.freq,mean(XX(:,:)),'filled','Color',co(1,:),'LineWidth',2);hold on
stem(ft_cell{1}.freq(pidx),mean(XX(:,pidx)),'filled','Color',co(4,:),'LineWidth',2);
xlim([1 40])
xticks(5:5:40);
a.YTick=[];
xlabel('Frequency (Hz)')
ylabel('Power')
title('Average all infants')
a2=axes('Position',[a.Position(1:2) .75*a.Position(3) .75*a.Position(4)]+[.35*a.Position(3) .2*a.Position(4) 0 0]);
cfg=[];
cfg.xlim = [5 5];    
cfg.layout = layout;
cfg.figure = a2;
cfg.comment='no';
cfg.style = 'straight';
cfg.colormap = [linspace(1,0.8392,100);...
    linspace(1,0.1529,100);...
    linspace(1,0.1569,100)]';
x=[];
for i=1:numel(ft_cell)
    [~,ia] = ismember(ft_cell{i}.label,layout.label);
    x(ia,:) = ft_cell{i}.powspctrm;
end
idx = find(sum(abs(x')));
ft = ft_cell{1};
ft.label = layout.label(idx);
ft.powspctrm = mean(x(idx,:));
ft_topoplotER(cfg,ft_cell{1});

%%
fn = './figures/figure2';
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=2;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
