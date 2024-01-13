
# experiments and subjects to be analyzed
experiments = ['ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3']  
subjects = [f"sub-{str(i).zfill(3)}" for i in range(1, 41)] # TODO: 40 subjects


# define triggers and stuff
delete_triggers = {'ERN': ['11', '12', '21', '22'], # stimulus triggers (only response locked analysis)
                   'LRP': ['11', '12', '21', '22'], # stimulus triggers (only response locked analysis)
                   'MMN': ['180'], # first stream of standards
                   'N170': ['201', '202'] + # responses (correct and incorrect)
                           [str(i) for i in list(range(101,141))] + # # scrambled faces 
                           [str(i) for i in list(range(141,181))], # scrambled cars            
                   'N2pc': ['201', '202'], # responses (correct and incorrect)
                   'N400': ['201', '202', # responses (correct and incorrect)
                            '111', '112', '121', '122'], # prime words
                   'P3': ['201', '202'], # responses (correct and incorrect)
                   }

# collate triggers so conditions are merged for decoding
conditions_triggers = {
    'ERN': {
        'correct': ['111','121','212','222'], # correct responses
        'incorrect' : ['112','122','211','221'],
    },
    'LRP': {
        'response_left': ['111','112','121','122'], # response left
        'response_right': ['211','212','221','222'], # response right
    },
    'N170': {
        'faces': [str(i) for i in list(range(1,41))], # faces
        'cars': [str(i) for i in list(range(41,81))], # cars
    },
    'N2pc': {
        'target_left': ['111','211','112','212'], # target left (target blue or pink, opening top or bottom)
        'target_right': ['121','122','221','222'], # target right (target blue or pink, opening top or bottom)
    },
    'N400': {
        'related': ['211', '212'], # related words
        'unrelated': ['221', '222'], # unrelated words
    },
    'MMN': {
        'standards': ['80'], # standards
        'deviants': ['70'], # deviants
    },
    'P3': {
        'standards': ['12','13','14','15',
                      '21','23','24','25',
                      '31','32','34','35',
                      '41','42','43','45',
                      '51','52','53','54'], # standards # TODO Future: one could also decode letters from standards
        'deviants': ['11','22','33','44','55'], # deviants
    },
}

# epoching windows
epoch_windows = {
    'ERN':  [-.6, .6],
    'LRP':  [-.8, .4],
    'MMN':  [-.4, .8],
    'N170': [-.4, .8],
    'N2pc': [-.4, .8],
    'N400': [-.4, .8],
    'P3':   [-.4, .8],
    }

# windows for baseline correction
baseline_windows = {
    '200ms': {
        'ERN':  (-.6, -.4),
        'LRP':  (-.8, -.6),
        'MMN':  (-.2, 0.),
        'N170': (-.2, 0.),
        'N2pc': (-.2, 0.),
        'N400': (-.2, 0.),
        'P3':   (-.2, 0.),
        },
    '400ms': {
        'ERN':  (-.6, -.2),
        'LRP':  (-.8, -.4),
        'MMN':  (-.2, 0.),
        'N170': (-.2, 0.),
        'N2pc': (-.2, 0.),
        'N400': (-.2, 0.),
        'P3':   (-.2, 0.),
        },
    }    

# define multiverse parameter space
multiverse_params = {
        'hpf': [None, 0.1, 0.5], # was 0.01, but the raw data has implicit hpf of 0.03 already
        'lpf': [None, 15, 45], # was 30
        'emc': [None, 'ica'],  # 'peak-to-peak', 
        'mus': [None, 'ica'], 
        'ref': [['FCz'], 'average', ['P9', 'P10']], # , 'mastoids'
        'base': [None, '200ms', '400ms'],
        # TODO: univariate noise normalization in baseline?: 
        # this can only be done in UNIVERSE, not MULTIVERSE, because value range would be much different from all other pipelines
        'det': [False, 'offset', 'linear'], # detrending should rather be combined with baseline correction, then det. is applied first. but it is again applied when loading data, so careful!
        'ar': [False, True], 
        }