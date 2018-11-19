class DataSet:
    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext):
        # ...
 
    def from_directory(self, dirpath, start_idx=0, limit=0, sort=None):
        return txt_filenames(dirpath, start_idx=start_idx, limit=limit, sort=sort)
 
    def next_batch(self, batch_size=None):
        idx_list = range(_start_idx, end_idx)
        txt_files = [_txt_files[i] for i in idx_list]
        wav_files = [x.replace('.txt', '.wav') for x in txt_files]
        # Load audio and text into memory
        (audio, text) = get_audio_and_transcript(
            txt_files,
            wav_files,
            _numcep,
            _numcontext)
