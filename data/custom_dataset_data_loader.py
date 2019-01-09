import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(config, filename):
    print(config)
    if config['dataset']['mode'] == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif config['dataset']['mode'] == 'haze':
        from data.haze_dataset import HazeDataset
        dataset = HazeDataset()
    elif config['dataset']['mode'] == 'rain':
        from data.rain_dataset import RainDataset
        dataset = RainDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % config['dataset_mode'])

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(config, filename)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, config, filename):
        BaseDataLoader.initialize(self, config, filename)
        self.dataset = CreateDataset(config, filename)
        batchSize = 1 if filename == 'test' else config['batch_size']
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=True,
            num_workers=int(config['num_workers']),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
