from torch.utils.data import Dataset
from utils.preprocess import *


class Cascades(Dataset):
    def __init__(self, dataset, max_seq_length, mode="train"):
        cascades, times = load_cascades(dataset, mode=mode)
        examples, examples_times = get_data_set(cascades, times,
                                                max_len=max_seq_length,
                                                mode=mode)
        self.examples, self.lengths, self.targets, self.masks, self.examples_times, self.targets_times = \
            prepare_sequences(examples, examples_times, max_len=max_seq_length, mode=mode)

    def __getitem__(self, index):
        examples = self.examples[index]
        lengths = self.lengths[index]
        targets = self.targets[index]
        masks = self.masks[index]
        examples_times = self.examples_times[index]
        targets_times = self.targets_times[index]
        return examples, lengths, targets, masks, examples_times, targets_times

    def __len__(self):
        return len(self.examples)
