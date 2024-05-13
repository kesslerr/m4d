%% eeglab
addpath('~/CoSMoMVPA/mvpa')
addpath('~/Matlabtoolboxes/eeglab')
eeglab nogui

%%
datapath = '../';
participants = readtable(sprintf('%s/participants.tsv',datapath),'FileType','text','Delimiter','\t');

stims = dir('./experiment/stimuli/stim*.png');
stims = unique({stims.name}');
assert(numel(stims)==200,'stimulus lookup failed')

subs = 1:100;
for subjectnr = subs
    fprintf('preprocessing sub-%03i\n',subjectnr);tic;
    
    %% get files
    mkdir(sprintf('%s/derivatives/cosmomvpa',datapath));
    
    cosmofn = sprintf('%s/derivatives/cosmomvpa/sub-%03i_cosmomvpa_clean_rawdata.mat',datapath,subjectnr);
    behavfn = sprintf('%s/sub-%03i/eeg/sub-%03i_task-fix_events.tsv',datapath,subjectnr,subjectnr);
    raw_filename = sprintf('%s/sub-%03i/eeg/sub-%03i_task-fix_eeg.set',datapath,subjectnr,subjectnr);
    
    if exist(raw_filename,'file')
        if ~exist(cosmofn,'file') || 0
            EEG_raw = pop_loadset(raw_filename);
    
            % re-reference
            EEG_raw = pop_reref(EEG_raw,[]);
        
            % clean data using the clean_rawdata plugin (recommended settings)
            EEG_raw = pop_clean_rawdata(EEG_raw,'FlatlineCriterion',5,'ChannelCriterion',0.85, ...
                'LineNoiseCriterion',4,'Highpass',[0.25 0.75] ,'BurstCriterion',20, ...
                'WindowCriterion',0.3,'BurstRejection','off','Distance','Euclidian', ...
                'WindowCriterionTolerances',[-Inf 7] ,'fusechanrej',1);
            EEG_raw = pop_reref(EEG_raw,[],'interpchan',[]);
        
            % high pass filter
            EEG_raw = pop_eegfiltnew(EEG_raw, .5,[]);
    
            % low pass filter
            EEG_raw = pop_eegfiltnew(EEG_raw, [],40);
        
            % find events
            events = arrayfun(@(x) str2double(strrep(x,'condition ','')),{EEG_raw.event.type});
            idx = events>0;
            onset = vertcat(EEG_raw.event(idx).latency);
            stimnum = events(idx)';
        
            [EEG_epoch,idx] = pop_epoch(EEG_raw,arrayfun(@num2str,1:200,'UniformOutput',false), [-0.100 0.800]);
            EEG_epoch = eeg_checkset(EEG_epoch);
        
            stimnum = stimnum(idx);
            onset = onset(idx);
        
            %% convert to cosmo
            ds = cosmo_flatten(permute(EEG_epoch.data,[3 1 2]),{'chan','time'},{{EEG_epoch.chanlocs.labels},EEG_epoch.times},2);
            ds.a.meeg=struct(); %or cosmo thinks it's not a meeg ds
            ds.sa=struct();
            ds.sa.onset = onset./EEG_epoch.srate;
            ds.sa.trialnr = cumsum(1+0*ds.sa.onset);
            ds.sa.stimnum = stimnum;
            ds.sa.stim = stims(ds.sa.stimnum);
            cosmo_check_dataset(ds,'meeg');
        
            %% save
            fprintf('saving...\n')
            save(cosmofn,'ds','-v7.3')
            fprintf('done\n')
        end
    end
end
