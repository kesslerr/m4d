import numpy as np
import mne

# delete all unnecessary triggers
def discard_triggers(raw, delete_triggers):
    """Delete all unnecessary triggers from the data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    delete_triggers : list of str
        List of triggers to delete.

    Returns
    -------
    raw : mne.io.Raw
        Raw data with deleted triggers.
    """
    raw.annotations.delete([i for i, x in enumerate(raw.annotations.description) if x in delete_triggers])
    return raw

# collate all triggers of the same conditions into the key of the dictionary
def rename_annotations(raw, conditions_triggers):
    """Collate all triggers of the same conditions into the key of the dictionary.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    conditions_triggers : dict
        Dictionary with conditions as keys and triggers as values.

    Returns
    -------
    raw : mne.io.Raw
        Raw data with collated triggers.
    """
    for key, value in conditions_triggers.items():
        raw.annotations.description = np.array(
            [
                key if i in value else i
                for i in raw.annotations.description
            ]
        )
    return raw

# make artificial EOG channels by combining existing channels
def recalculate_eog_signal(raw):
    """
    Recalculates the EOG (Electrooculogram) signal by creating HEOG (Horizontal EOG) and VEOG (Vertical EOG) channels.

    Args:
        raw (mne.io.Raw): The raw data containing the original EOG channels.

    Returns:
        mne.io.Raw: The raw data with the recalculated EOG channels.

    """
    #Create HEOG channel...
    heog_info = mne.create_info(['HEOG'], 256, "eog")
    heog_data = raw['HEOG_left'][0]-raw['HEOG_right'][0]
    heog_raw = mne.io.RawArray(heog_data, heog_info)
    #...and VOEG
    veog_info = mne.create_info(['VEOG'], 256, "eog")
    veog_data = raw['VEOG_lower'][0]-raw['FP2'][0]
    veog_raw = mne.io.RawArray(heog_data, veog_info)
    #Append them to the data
    raw.add_channels([heog_raw, veog_raw],True)
    # delete original EOG channels
    raw.drop_channels([ 'HEOG_left', 'HEOG_right', 'VEOG_lower'])
    
    return raw

# set the Luck et al montage
def set_montage(raw):
    """
    Sets the montage for the Luck et al. raw data.

    Parameters:
    raw (mne.io.Raw): The raw data to set the montage for.

    Returns:
    mne.io.Raw: The raw data with the montage set.
    """
    # rename channels so they match with templates
    raw.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))

    mont1020 = mne.channels.make_standard_montage('standard_1020')
    # Choose what channels you want to keep
    kept_channels = raw.ch_names

    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if (channel in kept_channels)] # or (channel in add_channels)
    mont = mont1020.copy()

    # Keep only the desired channels
    mont.ch_names = [mont1020.ch_names[x]for x in ind]
    kept_channel_info = [mont1020.dig[x + 3] for x in ind]

    # Keep the first three rows as they are the fiducial points information
    mont.dig = mont1020.dig[:3] + kept_channel_info

    # plot
    #mont.plot()

    # Apply the montage
    raw.set_montage(mont)
    
    return raw