
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
        'incorrect' : ['112','122','211','221'], # incorrect responses
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
        'ref': [['Cz'], 'average', ['P9', 'P10']], # 'Cz' instead of FCz because some experiments have FCz as electrode of interest (would be 0 else)
        'hpf': [None, 0.1, 0.5], # was 0.01, but the raw data has implicit hpf of 0.03 already
        'lpf': [None, 6, 20, 45], # 6 Hz for alpha exclusion, Bae and Luck 2018, 2019a, 2019b (Ref in Bae 2021)
        'emc': [None, 'ica'],  # 'peak-to-peak', 'regression'
        'mus': [None, 'ica'], 
        'base': ['200ms', '400ms'], # "None" is ommitted, makes no sense not to detrend
        # TODO: univariate noise normalization in baseline?: 
        # this can only be done in UNIVERSE, not MULTIVERSE, because value range would be much different from all other pipelines
        'det': ['offset', 'linear'], # "False" is ommitted, makes no sense not to detrend
        'ar': [False, True], 
        }

channels_of_interest = {
        'ERN': ['FCz'], # not compatible with some forking paths
        'LRP': ['C3', 'C4'],
        'MMN': ['FCz'], # not compatible with some forking paths
        'N170': ['PO8'],
        'N2pc': ['PO7', 'PO8'],
        'N400': ['CPz'],
        'P3': ['Pz'],
        }

luck_references = {
        'ERN': ['P9', 'P10'],
        'LRP': ['P9', 'P10'],
        'MMN': ['P9', 'P10'],
        'N170': 'average',
        'N2pc': ['P9', 'P10'],
        'N400': ['P9', 'P10'],
        'P3': ['P9', 'P10'],
        }

luck_forking_paths = { # these are not really the same, but some steps are comparable
        'ERN': "['P9', 'P10']_0.1_None_ica_ica_200ms_offset_True",
        'LRP': "['P9', 'P10']_0.1_None_ica_ica_200ms_offset_True",
        'MMN': "['P9', 'P10']_0.1_None_ica_ica_200ms_offset_True",
        'N170': "average_0.1_None_ica_ica_200ms_offset_True",
        'N2pc': "['P9', 'P10']_0.1_None_ica_ica_200ms_offset_True",
        'N400': "['P9', 'P10']_0.1_None_ica_ica_200ms_offset_True",
        'P3': "['P9', 'P10']_0.1_None_ica_ica_200ms_offset_True",
        }

# # difference waveforms, define the partners of subtraction
# # not use atm, as more easy to compute in the evoked.py script
# difference_waveforms = {
#     'ERN': {
#         'incorrect-correct': ['incorrect', 'correct'],
#     },
#     'LRP': {
#         'response_left-response_right': ['response_left', 'response_right'],
#         'response_right-response_left': ['response_right', 'response_left'],
#     },
#     'N170': {
#         'faces-cars': ['faces', 'cars'],
#     },
#     'N2pc': {
#         'target_left-target_right': ['target_left', 'target_right'],
#         'target_right-target_left': ['target_right', 'target_left'],
#     },
#     'N400': {
#         'unrelated-related': ['unrelated', 'related'],
#     },
#     'MMN': {
#         'deviants-standards': ['deviants', 'standards'],
#     },
#     'P3': {
#         'deviants-standards': ['deviants', 'standards'],
#     },
# }