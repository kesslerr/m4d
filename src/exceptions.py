import numpy as np

# for sub-008, experiment = "N2pc", the trigger '-99' should be 'target_left' instead

def exception_pre_preprocessing_annotations(experiment, subject, raw):
    
    if subject == "sub-008" and experiment == "N2pc":
        raw.annotations.description = np.array(
            [
                'target_left' if i == '-99' else i
                for i in raw.annotations.description
            ]
        )
    
    return raw
    
