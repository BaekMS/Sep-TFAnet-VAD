from base import BaseDataLoader
from data_loader.datasets import Realistic_dataset
from data_loader.datasets import Whamr_dataset

class Our_DataLoader(BaseDataLoader):
    def __init__(self, csv_file, csd_labels_freq, batch_size, type_dataset, shuffle=True, validation_split=0.0, num_workers=1):
        self.csv_file = csv_file
        if type_dataset == "whamr":  
            self.dataset = Whamr_dataset(csv_file, csd_labels_freq)
        elif type_dataset == "realistic_data":
            self.dataset = Realistic_dataset(csv_file, csd_labels_freq)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
        
