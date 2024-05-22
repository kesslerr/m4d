
%%
addpath('~/CoSMoMVPA/mvpa')

%%
for subjectnr = 1:100

    fprintf('decoding sub-%03i\n',subjectnr);tic;

    %% get files
    datapath = '../';
    cosmofn = sprintf('%s/derivatives/cosmomvpa/sub-%03i_cosmomvpa_clean_rawdata.mat',datapath,subjectnr);

    if exist(cosmofn,'file')
        try
            %%
            load(cosmofn)

            % baseline correct
            ds = cosmo_meeg_baseline_correct(ds,[-.1 0],'absolute');

            % take only posterior channels
            ds = cosmo_slice(ds,cellfun(@(x) ~ismember('F',x) && ~ismember('T',x),ds.a.fdim.values{1}(ds.fa.chan)),2);

            % pca-transform channel data
            dst = cosmo_dim_transpose(ds,'time');
            [~,dst.samples] = pca(dst.samples);
            ds = cosmo_dim_transpose(dst,'time',2,1);
            ds.samples = double(ds.samples);

            ds.sa.targets = ds.sa.stimnum;
            ds = cosmo_fx(ds,@(x) mean(x,1),{'targets'});

            cosmo_check_dataset(ds);
            nh = cosmo_interval_neighborhood(ds,'time','radius',0);
            measure = @cosmo_dissimilarity_matrix_measure;
            ma = {};
            ma.metric = 'correlation';
            ma.center_data = 1;

            res_rdm = cosmo_searchlight(ds,nh,measure,ma);

            rdm = nan(size(res_rdm.samples,2),200,200);
            
            for i = 1:size(res_rdm.samples,1)
                rdm(:,res_rdm.sa.targets1(i),res_rdm.sa.targets2(i)) = res_rdm.samples(i,:);
                rdm(:,res_rdm.sa.targets2(i),res_rdm.sa.targets1(i)) = res_rdm.samples(i,:);
            end

            save(sprintf('../derivatives/results/sub-%03i_rdm.mat',subjectnr),"res_rdm","rdm")
        catch error
            disp(error)
        end
    end
end