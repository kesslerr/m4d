%%
addpath('~/CoSMoMVPA/mvpa/')
addpath('~/Repository/CommonFunctions/matplotlib/')

stats = {};

%% load adult data
fprintf('loading adult data\n');
adult_data=load('adultRDMs.mat');
stats.adult_timevec = adult_data.timevect;

%% load infant data
fprintf('loading infant data\n');
fns = dir('../derivatives/results/sub-*_rdm.mat');
infant_data={};cc=clock();mm='';
for f=1:numel(fns)
    fn = fullfile(fns(f).folder,fns(f).name);
    x=load(fn);
    infant_data.grouprdm_5hz(f,:,:,:) = x.rdm;
    stats.infant_timevec = x.res_rdm.a.fdim.values{1};
    stats.sub_id{f} = fns(f).name;
    mm=cosmo_show_progress(cc,f/numel(fns),sprintf('%i/%i loading ../results/%s\n',f,numel(fns),fns(f).name),mm);
end
fprintf('finished\n');

%% time time correlations and permutation test
fprintf('time time correlation\n');
loweridx = find(tril(ones(200),-1));
stats.nboot=1000;
stats.clusterformingthreshold = .05;
X = squeeze(nanmean(infant_data.grouprdm_5hz));
Y = squeeze(mean(adult_data.grouprdm_5hz(:,:,loweridx)))';
stats.clustermeasure = 'sum';
cc = clock();mm='';stats.bootclustermeasure=[];
for k=1:stats.nboot
    rng(k)
    XX = X;
    px = randperm(200);
    XX = XX(:,px,px);
    [r,p_uncorrected] = corr(XX(:,loweridx)',Y,'type','Spearman','tail','right');
    p_threshold = p_uncorrected < stats.clusterformingthreshold;

    clusterresult = bwconncomp(p_threshold); % find clusters
    clusters = clusterresult.PixelIdxList;
    if strcmp(stats.clustermeasure,'size')
        stats.bootclustermeasure(k) = max([0 cellfun(@numel,clusters)]); % store largest cluster size
    elseif strcmp(stats.clustermeasure,'sum')
        stats.bootclustermeasure(k) = max([0 cellfun(@(x) sum(r(x)), clusters)]); % store largest cluster sum
    end
    mm = cosmo_show_progress(cc,k/stats.nboot,sprintf('%i/%i',k,stats.nboot),mm);
end
XX = X;
[stats.R,stats.p_uncorrected] = corr(XX(:,loweridx)',Y,'type','Spearman','tail','right');
stats.p_threshold = stats.p_uncorrected < stats.clusterformingthreshold;
stats.clusterresult = bwconncomp(stats.p_threshold); % find clusters
stats.clusters = stats.clusterresult.PixelIdxList;
if strcmp(stats.clustermeasure,'size')
    stats.clustermeasure = cellfun(@numel,stats.clusters);
elseif strcmp(stats.clustermeasure,'sum')
    stats.clustermeasure = cellfun(@(x) sum(stats.R(x)), stats.clusters);
end
stats.significant_cluster_idx = stats.clustermeasure>prctile(stats.bootclustermeasure,95);
stats.significant_clusters = stats.clusters(stats.significant_cluster_idx);

stats.thresholded_cluster_map = zeros(size(stats.R));
stats.thresholded_cluster_map(vertcat(stats.significant_clusters{:})) = 1;

%% correlations with models
stats.models = [];
stats.models(:,1) = pdist(ceil((1:200)/100)','jaccard');
stats.models(:,2) = pdist(ceil((1:200)/20)','jaccard');
stats.models(:,3) = pdist(ceil((1:200)/4)','jaccard');

stats.modelnames = {'animacy','category','object'};
[stats.r_adult,stats.p_adult] = corr(squeeze(mean(adult_data.grouprdm_5hz(:,:,loweridx)))',stats.models,'type','Spearman','tail','right');
[stats.r_infant,stats.p_infant] = corr(squeeze(nanmean(infant_data.grouprdm_5hz(:,:,loweridx)))',stats.models,'type','Spearman','tail','right');

%% 
save('../derivatives/results/rsa_stats.mat','stats','-v7.3')

