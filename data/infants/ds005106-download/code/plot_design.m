%%
addpath('~/Repository/CommonFunctions/matplotlib/')

%% read im
I={};N={};alph={};
for i=1:200
    d=dir(sprintf('./experiment/stimuli/stim%03i_*',i));
    fn = d(1).name;
    [im,~,a]=imread(fullfile(d(1).folder, fn));
    I{i} = im;
    alph{i} = a;
    N{i} = fn;
    i
end

%% figure_stimulus_montage
f=figure(1);clf
f.Position(3:4) = [1285 1000];

aw = 40; % image size
left=10;
bottom=280;
bufferw=20;
bufferh=30;
n=20;

% use these colours:
cat1col = [0.2080    0.7187    0.4729];
cat2col = [0.1906    0.4071    0.5561];
cat3col = [0.2670    0.0049    0.3294];

co = tab10();
cat1col = co(1,:);
cat2col = co(2,:);
cat3col = co(3,:);

categorylabels = {'Animacy' 'Category' 'Object'};

annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+80 bottom+10*aw+9*bufferh 20 0],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+80 bottom+5*aw+5*bufferh 20 0],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+80 bottom+5*aw+4*bufferh 20 0],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+80 bottom 20 0],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+100 bottom 0 5*aw+4*bufferh],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+100 bottom+5*aw+5*bufferh 0 5*aw+4*bufferh],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+100 bottom+7.5*aw+7*bufferh 20 0],'LineWidth',2)
annotation('line','Units','Pixels','Position',[left+20*aw+5*bufferw+100 bottom+2.5*aw+2*bufferh 20 0],'LineWidth',2)
annotation('textbox','Units','Pixels','Position',[left+20*aw+5*bufferw+120 bottom+7.5*aw+7*bufferh-10 100 20],'Color',cat1col,'LineStyle','none','String','Animate','FontSize',20,'VerticalAlignment','middle');
annotation('textbox','Units','Pixels','Position',[left+20*aw+5*bufferw+120 bottom+2.5*aw+2*bufferh-10 100 20],'Color',cat1col,'LineStyle','none','String','Inanimate','FontSize',20,'VerticalAlignment','middle');

for i=1:200
    row = (200/n)-ceil(i/n);
    col = mod(i-1,n);
    a = axes('Units','pixels','Position',[left+col*aw+floor(col/4)*bufferw bottom+row*(aw+bufferh) aw aw],'Visible','off');
    a.XLim = [.5 length(I{i})+.5];
    a.YLim = a.XLim;
    
    fp = strsplit(N{i},'_');
    h = imshow(I{i});
    set(h, 'AlphaData', alph{i});
    if col+1==n
         text(300,mean(a.YLim),fp{3},'Color',cat2col,'FontSize',20,'HorizontalAlignment','left','VerticalAlignment','middle') % basic category label
    end
    if mod(col,4)==2
        text(a.XLim(1),a.YLim(1),fp{4},'Color',cat3col,'FontSize',16,'HorizontalAlignment','center','VerticalAlignment','bottom') % object label
    end
end

% legend for category levels
annotation('textbox','Units','Pixels','Position',[5 bottom+10*(aw+bufferh)-5 890 25],'BackgroundColor',cat3col,'Color','w','String','OBJECT','LineStyle','none','HorizontalAlignment','center','FontSize',16,'VerticalAlignment','middle') 
annotation('textbox','Units','Pixels','Position',[895 bottom+10*(aw+bufferh)-5 110 25],'BackgroundColor',cat2col,'Color','w','String','CATEGORY','LineStyle','none','HorizontalAlignment','center','FontSize',16,'VerticalAlignment','middle')
annotation('textbox','Units','Pixels','Position',[1005 bottom+10*(aw+bufferh)-5 155 25],'BackgroundColor',cat1col,'Color','w','String','SUPERORDINATE','LineStyle','none','HorizontalAlignment','center','FontSize',16,'VerticalAlignment','middle')

% timeline
rng(10)% set the randomiser state

% make 2 image streams of examples - get 10 (1-200) and one target (0) and
% leave space for fixation (-3) and ...(-1) and response screen (-2)
shufims = randperm(200);
shufims = shufims(randperm(length(shufims)));
stream = [-3 10 0 103 0 134 0 -1 91 0 42 0 190 0];
imnums = [0 1 0 2 0 3 0 0 198 0 199 0 200 0 0];

aw = 80; % image size
left=20;
bottom=115;
bufferw=1;
bufferh=200; % gap between two screens
timeliney = 60;
epyoffset = 120;
n=12;

for i=1:length(stream)
    row = 0;
    col = i-1;
    
    a = axes('Units','pixels','Position',[left+col*(aw+bufferw) bottom+row*(aw+bufferh) aw aw],'Visible','off');
    a.XLim = [.5 length(stream)+.5];
    a.YLim = a.XLim;
    
    imagepos = [a.Position(1) a.Position(2) aw aw]; % square
    labelpos = [a.Position(1) a.Position(2)+aw+15 aw 20]; % above image
    timingpos = [a.Position(1) a.Position(2)-40 aw 20]; % below image
    line1 = [a.Position(1) a.Position(2)-10 aw 0]; % across
    line2 = [a.Position(1) a.Position(2)-10 0 10]; % start mark
    line3 = [a.Position(1)+aw a.Position(2)-10 0 10]; % end mark
    

    % draw fixation cross
    if stream(i) == -3
        
        % % draw cross
        % annotation('textbox','Units','Pixels','Position',imagepos,'LineStyle','none','String', ...
        %     '+','FontSize',20,'VerticalAlignment','middle','HorizontalAlignment','center');
        
        % draw image
        [im,~,ima] = imread('./experiment/Fixation.png');
        h = imshow(im);
        set(h, 'AlphaData', ima);

        % line under stimulus
        annotation('line','Units','Pixels','Position',line1,'LineWidth',2);
        annotation('line','Units','Pixels','Position',line2,'LineWidth',2);
        annotation('line','Units','Pixels','Position',line3,'LineWidth',2);
        
        % time label
        annotation('textbox','Units','Pixels','Position',timingpos,'LineStyle','none','String', ...
            'Until fixation','FontSize',16,'VerticalAlignment','middle','HorizontalAlignment','center');

        if i==1
            % draw full timeline arrow
            annotation('arrow',[a.Position(1)/f.Position(3) (a.Position(1)+length(stream)*(aw+bufferw))/f.Position(3)],[(a.Position(2)-80)/f.Position(4) (a.Position(2)-80)/f.Position(4)],'LineWidth',3);
            % label whole sequence - near timeline and upper left
            annotation('textbox','Units','Pixels','Position',[a.Position(1)+(length(stream)-3)*(aw+bufferw) (a.Position(2)-105) aw*3 20],'LineStyle','none','String', ...
                '~40 second sequence','FontSize',20,'VerticalAlignment','middle','HorizontalAlignment','right');
            annotation('textbox','Units','Pixels','Position',[a.Position(1) a.Position(2)+aw+45 aw*3 20],'LineStyle','none','String',...
                'Rapid sequence design','FontSize',20,'VerticalAlignment','middle','FontWeight','bold');
        end
        
    elseif stream(i) >= 0 % one of the 200 objects
        
        % draw image
        if stream(i) > 0
            h = imshow(I{stream(i)});
            set(h, 'AlphaData', alph{stream(i)});
            % image label
            annotation('textbox','Units','Pixels','Position',labelpos,'LineStyle','none','String',...
                num2str(imnums(i)),'FontSize',20,'VerticalAlignment','middle','HorizontalAlignment','center');
        end
        
        % line under stimulus
        annotation('line','Units','Pixels','Position',line1,'LineWidth',2);
        annotation('line','Units','Pixels','Position',line2,'LineWidth',2);
        annotation('line','Units','Pixels','Position',line3,'LineWidth',2);

        % time label
        annotation('textbox','Units','Pixels','Position',timingpos,'LineStyle','none','String',...
            '100ms','FontSize',16,'VerticalAlignment','middle','HorizontalAlignment','center');       
    % draw '...'
    elseif stream(i) == -1 
        annotation('line','Units','Pixels','Position',[a.Position(1)+20 a.Position(2)+40 aw-40 0],'LineWidth',2,'LineStyle',':')
    % draw blank       
    elseif stream(i) == 0
        annotation('line','Units','Pixels','Position',[a.Position(1)+20 a.Position(2)+40 aw-40 0],'LineWidth',2,'LineStyle','none')
    end        
end



%%
fn = './figures/figure1';
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=2;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
