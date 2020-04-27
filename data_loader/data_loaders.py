import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image
from sklearn.preprocessing import LabelEncoder


class HairCAMDataset(Dataset):
    '''
    Dataset for accessing Hair-CAM data samples
    '''

    def __init__(self, csv_file, data_dir, train=True, transform=None):
        '''
        :param csv_file: String, csv file with data labels and train/valid splits
        :param data_dir: String, data directory
        :param train: Bool, whether to access train/validation samples
        :param transform: Callable, (optional) transform to be applied to a sample
        '''
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(data_dir, csv_file), nrows=500)
        self.transform = transform
        self.train = train

        # load train data only
        if self.train is True:
            self.data = self.data.loc[self.data.valid == 0].reset_index(drop=True)
        # load valid data only
        if self.train is False:
            self.data = self.data.loc[self.data.valid == 1].reset_index(drop=True)

        # label encode target
        self.target = LabelEncoder().fit_transform(self.data.hair_type)

    # override __len__ and __getitem__ methods
    def __len__(self):
        '''
        Allow using `len(dataset)` to get size of dataset instance
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Allow indexing using `dataset[i]` to access ith data sample

        :param idx: Integer, index of data sample in dataset
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_n = os.path.join(self.data_dir, self.data.loc[idx, 'image'])
        img = Image.open(fp=img_n)
        label = self.data.loc[idx, 'hair_type']
        target = self.target[idx]
        sample = {
            'image': img,
            'label': label,
            'target': target
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample['image'], sample['target']


class HairCAMDataLoader(DataLoader):
    '''
    Data loader for Hair-CAM data samples
    '''

    def __init__(self, csv_file, data_dir, batch_size, shuffle=True,
                 num_workers=1, trsfm=None, training=True):
        '''
        :param csv_file: String, csv file with data labels and train/valid splits
        :param data_dir: String, data directory
        :param batch_size: Int, Number of data samples in training batch
        :param shuffle: Bool, (optional) Whether to reshuffle the data every epoch
        :param num_workers: Int, number of subprocesses to use for data loading
        :param training: Bool, whether to access train/validation samples
        :param transform: Callable, (optional) transform to be applied to a sample
        '''
        # initialize dataset
        self.dataset = HairCAMDataset(
            csv_file, data_dir, train=training, transform=trsfm)

        # call parent's (`DataLoader`) ``__init__`` method
        super().__init__(self.dataset, batch_size, shuffle=shuffle,
                         sampler=None, num_workers=num_workers)


def get_haircam_data_loaders(csv_file, data_dir, batch_size, shuffle=True,
                             num_workers=1, trsfm=None):
    '''
    Helper function to get train/valid data loaders

    :param csv_file: String, name of csv file with data labels and train/valid splits
    :param data_dir: String, data directory
    :param batch_size: Int, Number of data samples in training batch
    :param shuffle: Bool, (optional) Whether to reshuffle the data every epoch
    :param num_workers: Int, (optional) number of subprocesses to use for data loading
    :param training: Bool, whether to access train/validation samples
    :param transform: Callable, (optional) transform to be applied to a sample
    '''

    # image transforms
    trsfm = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_loader = HairCAMDataLoader(csv_file=csv_file, data_dir=data_dir,
                                     batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, trsfm=trsfm,
                                     training=True)

    valid_loader = HairCAMDataLoader(csv_file=csv_file, data_dir=data_dir,
                                     batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, trsfm=trsfm,
                                     training=False)
    return train_loader, valid_loader
