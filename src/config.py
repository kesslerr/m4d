import pandas as pd
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# experiments and subjects to be analyzed
experiments = ['ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3']  
subjects = [f"sub-{str(i).zfill(3)}" for i in range(1, 41)] # TODO: 40 subjects

# cichy / paperclip
cichy_exclude = [4,8,9,15,25,30,32,40]
cichy_subjects_infants = [f"sub-{str(i).zfill(2)}" for i in range(1, 49) if i not in cichy_exclude] 
cichy_subjects_adults = [f"sub-{str(i).zfill(2)}" for i in range(1, 21)]

# child mind institute
subjects_mipdb = pd.read_csv('data/mipdb/participants.csv')['ID'].tolist()

# mipdb age groups and their corresponding participants
subjects_mipdb_dem = pd.read_csv('data/mipdb/participants.csv')
age_groups = {"6-9": [6,7,8,9],
              "10-11": [10,11],
              "12-13": [12,13],
              "14-17": [14,15,16,17],
              "18-25": [18,19,20,21,22,23,24,25],
              }

groups_subjects_mipdb = {}
for group, ages in age_groups.items():
    groups_subjects_mipdb[group] = subjects_mipdb_dem[subjects_mipdb_dem['Age'].isin(ages)]['ID'].tolist()
    


# define triggers and stuff
delete_triggers = { # ERPCORE
                   'ERN': [], # '11', '12', '21', '22' stimulus triggers (only response locked analysis)
                   'LRP': ['11', '12', '21', '22'], # stimulus triggers (only response locked analysis)
                   'MMN': ['180', # first stream of standards
                           '1', '4'], # these triggers were found with no reference in the data (1-2 occurences per participant), TODO: write Luck 
                   'N170': ['201', '202'] + # responses (correct and incorrect)
                           [str(i) for i in list(range(101,141))] + # # scrambled faces 
                           [str(i) for i in list(range(141,181))], # scrambled cars            
                   'N2pc': ['201', '202'], # responses (correct and incorrect)
                   'N400': ['201', '202', # responses (correct and incorrect)
                            '111', '112', '121', '122'], # prime words
                   'P3': ['201', '202'], # responses (correct and incorrect)
                   
                   # Cichy
                   'paperclip': ['New Segment/', 'Stimulus/S129', 'Stimulus/S200', 'Stimulus/S222','Stimulus/S244'], # found in adults, probably paperclip stimuli (for blinks) and responses to it
                   
                   # infants
                   'RSVP': ["-1", "-2"]
                   }

# collate triggers so conditions are merged for decoding
conditions_triggers = {
    'ERN': {
        'correct': ['111','121','212','222'], # correct responses
        'incorrect' : ['112','122','211','221'], # incorrect responses
        'stimulus': ['11','12','21','22'], # stimulus triggers (new, for stimulus locked baseline correction)
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
    'paperclip': {
        'toy': ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3',
                'Stimulus/S  4', 'Stimulus/S  5', 'Stimulus/S  6', 'Stimulus/S  7',
                'Stimulus/S  8', 'Stimulus/S  9', 'Stimulus/S 10', 'Stimulus/S 11',
                'Stimulus/S 12', 'Stimulus/S 13', 'Stimulus/S 14', 'Stimulus/S 15',
                'Stimulus/S 16', 'Stimulus/S 17', 'Stimulus/S 18', 'Stimulus/S 19',
                'Stimulus/S 20', 'Stimulus/S 21', 'Stimulus/S 22', 'Stimulus/S 23',
                'Stimulus/S 24', 'Stimulus/S 25', 'Stimulus/S 26', 'Stimulus/S 27',
                'Stimulus/S 28', 'Stimulus/S 29', 'Stimulus/S 30', 'Stimulus/S 31',
                'Stimulus/S 32'], 
        'body': ['Stimulus/S 33', 'Stimulus/S 34', 'Stimulus/S 35',
                'Stimulus/S 36', 'Stimulus/S 37', 'Stimulus/S 38', 'Stimulus/S 39',
                'Stimulus/S 40', 'Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 43',
                'Stimulus/S 44', 'Stimulus/S 45', 'Stimulus/S 46', 'Stimulus/S 47',
                'Stimulus/S 48', 'Stimulus/S 49', 'Stimulus/S 50', 'Stimulus/S 51',
                'Stimulus/S 52', 'Stimulus/S 53', 'Stimulus/S 54', 'Stimulus/S 55',
                'Stimulus/S 56', 'Stimulus/S 57', 'Stimulus/S 58', 'Stimulus/S 59',
                'Stimulus/S 60', 'Stimulus/S 61', 'Stimulus/S 62', 'Stimulus/S 63',
                'Stimulus/S 64'], 
        'houses': ['Stimulus/S 65', 'Stimulus/S 66', 'Stimulus/S 67',
                'Stimulus/S 68', 'Stimulus/S 69', 'Stimulus/S 70', 'Stimulus/S 71',
                'Stimulus/S 72', 'Stimulus/S 73', 'Stimulus/S 74', 'Stimulus/S 75',
                'Stimulus/S 76', 'Stimulus/S 77', 'Stimulus/S 78', 'Stimulus/S 79',
                'Stimulus/S 80', 'Stimulus/S 81', 'Stimulus/S 82', 'Stimulus/S 83',
                'Stimulus/S 84', 'Stimulus/S 85', 'Stimulus/S 86', 'Stimulus/S 87',
                'Stimulus/S 88', 'Stimulus/S 89', 'Stimulus/S 90', 'Stimulus/S 91',
                'Stimulus/S 92', 'Stimulus/S 93', 'Stimulus/S 94', 'Stimulus/S 95',
                'Stimulus/S 96'], 
        'faces': ['Stimulus/S 97', 'Stimulus/S 98',
                'Stimulus/S 99', 'Stimulus/S100', 'Stimulus/S101', 'Stimulus/S102',
                'Stimulus/S103', 'Stimulus/S104', 'Stimulus/S105', 'Stimulus/S106',
                'Stimulus/S107', 'Stimulus/S108', 'Stimulus/S109', 'Stimulus/S110',
                'Stimulus/S111', 'Stimulus/S112', 'Stimulus/S113', 'Stimulus/S114',
                'Stimulus/S115', 'Stimulus/S116', 'Stimulus/S117', 'Stimulus/S118',
                'Stimulus/S119', 'Stimulus/S120', 'Stimulus/S121', 'Stimulus/S122',
                'Stimulus/S123', 'Stimulus/S124', 'Stimulus/S125', 'Stimulus/S126',
                'Stimulus/S127', 'Stimulus/S128'],
    },
    
}


# Groot. infants
category_triggers = {
    'aquatic':   [str(i) for i in range(1,21)],
    'bird':      [str(i) for i in range(21,41)],
    'human':     [str(i) for i in range(41,61)],
    'insect':    [str(i) for i in range(61,81)],
    'mammal':    [str(i) for i in range(81,101)],
    'clothing':  [str(i) for i in range(101,121)],
    'fruits':    [str(i) for i in range(121,141)],
    'furniture': [str(i) for i in range(141,161)],
    'plants':    [str(i) for i in range(161,181)],
    'tools':     [str(i) for i in range(181,201)],
}
supraordinate_triggers = {
    'animate':   [str(i) for i in range(1,101)],
    'inanimate': [str(i) for i in range(101,201)],
}

# epoching windows
epoch_windows = {
    # ERPCORE
    'ERN':  [-.6, .6],
    'LRP':  [-.8, .4],
    'MMN':  [-.4, .8],
    'N170': [-.4, .8],
    'N2pc': [-.4, .8],
    'N400': [-.4, .8],
    'P3':   [-.4, .8],
    # MIPDB
    'MIPDB': [-.8, .4],
    # infants
    'RSVP': [-.4, .8],
    }

# windows for baseline correction
baseline_windows = {
    '200ms': { # correspond to Kappenmann et al.
        'ERN':  (-.4, -.2), 
        'LRP':  (-.6, -.4), # TODO: also change in targets, now the endpoint is matched between paths, TODO also change in manuscript
        'MMN':  (-.2, 0.),
        'N170': (-.2, 0.),
        'N2pc': (-.2, 0.),
        'N400': (-.2, 0.),
        'P3':   (-.2, 0.),
        'MIPDB': [-.8, -.6],
        'RSVP': [-.2, 0.],
        },
    '400ms': {
        'ERN':  (-.6, -.2),
        'LRP':  (-.8, -.4),
        'MMN':  (-.4, 0.),
        'N170': (-.4, 0.),
        'N2pc': (-.4, 0.),
        'N400': (-.4, 0.),
        'P3':   (-.4, 0.),
        'MIPDB': [-.8, -.4],
        'RSVP': [-.4, 0.],
        },
    }    

# for average accuracy estimataion
baseline_end = {
    "ERN": -0.2, 
    "LRP": -0.4, # caution, this is after the 200ms version of the baseline to be fair with the 400ms version
    "MMN": 0, 
    "N170": 0, 
    "N2pc": 0, 
    "N400": 0, 
    "P3": 0
    }


decoding_windows = epoch_windows.copy()

# # if potential baseline should not be included
# decoding_windows = {
#     # ERPCORE
#     'ERN':  [-.2, .6],
#     'LRP':  [-.4, .4],
#     'MMN':  [.0, .8],
#     'N170': [.0, .8],
#     'N2pc': [.0, .8],
#     'N400': [.0, .8],
#     'P3':   [.0, .8],
#     # MIPDB
#     'MIPDB': [-.4, .4],
#     }

# define multiverse parameter space
multiverse_params = {
        'ref': [['Cz'], 'average', ['P9', 'P10']], # 'Cz' instead of FCz because some experiments have FCz as electrode of interest (would be 0 else)
        'hpf': [None, 0.1, 0.5], # was 0.01, but the raw data has implicit hpf of 0.03 already
        'lpf': [None, 6, 20, 45], # 6 Hz for alpha exclusion, Bae and Luck 2018, 2019a, 2019b (Ref in Bae 2021)
        'emc': [None, 'ica'],  # 'peak-to-peak', 'regression'
        'mus': [None, 'ica'], 
        'det': [None, 'linear'], 
        'base': [None, '200ms', '400ms'], # "None" is ommitted, makes no sense not to detrend
        # TODO: univariate noise normalization in baseline?: 
        # this can only be done in UNIVERSE, not MULTIVERSE, because value range would be much different from all other pipelines
        'ar': [False, 'int', 'intrej'], 
        }

# replace special characters in the multiverse parameter space
translation_table = str.maketrans("", "", "[],' ")

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
        'ERN': "ica_ica_None_0.1_P9P10_None_200ms_int",
        'LRP': "ica_ica_None_0.1_P9P10_None_200ms_int",
        'MMN': "ica_ica_None_0.1_P9P10_None_200ms_int",
        'N170': "ica_ica_None_0.1_average_None_200ms_int",
        'N2pc': "ica_ica_None_0.1_P9P10_None_200ms_int",
        'N400': "ica_ica_None_0.1_P9P10_None_200ms_int",
        'P3': "ica_ica_None_0.1_P9P10_None_200ms_int",
        'MIPDB': "ica_ica_None_0.1_P9P10_None_200ms_int",
        }




# difference waveforms, define the partners of subtraction
# not use atm, as more easy to compute in the evoked.py script
contrasts = {
    'ERN': {
        'incorrect - correct': ['incorrect', 'correct'],
    },
    'LRP': {
        'response_right - response_left': ['response_right', 'response_left'], # corresponds to what is seen at C3 (right - left / contra - ipsi)
        'response_left - response_right': ['response_left', 'response_right'], # corresponds to what is seen at C4 (left - right / contra - ipsi)
    },
    'MMN': {
        'deviants - standards': ['deviants', 'standards'],
    },
    'N170': {
        'faces - cars': ['faces', 'cars'],
    },
    'N2pc': {
        'target_right - target_left': ['target_right', 'target_left'], # what is seen at PO7 (right - left / contra - ipsi)
        'target_left - target_right': ['target_left', 'target_right'], # what is seen at PO8 (left - right / contra - ipsi)
    },
    'N400': {
        'unrelated - related': ['unrelated', 'related'],
    },
    'P3': {
        'deviants - standards': ['deviants', 'standards'],
    },
}

contrasts_combined = {
    'ERN': {
        'incorrect - correct': ['incorrect', 'correct'],
    },
    'LRP': {
        'contralateral - ipsilateral': ['contralateral', 'ipsilateral'], 
    },
    'MMN': {
        'deviants - standards': ['deviants', 'standards'],
    },
    'N170': {
        'faces - cars': ['faces', 'cars'],
    },
    'N2pc': {
        'contralateral - ipsilateral': ['contralateral', 'ipsilateral'], 
    },
    'N400': {
        'unrelated - related': ['unrelated', 'related'],
    },
    'P3': {
        'deviants - standards': ['deviants', 'standards'],
    },
}