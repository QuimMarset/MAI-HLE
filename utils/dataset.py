from torch.utils.data import Dataset, DataLoader



class CustomDateset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __getitem__(self, index):
        return self.data[index], self.labels[index]


    def __len__(self):
        return self.labels.shape[0]



def create_data_loader(data, labels, batch_size, shuffle):
    dataset = CustomDateset(data, labels)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )
    return data_loader


def create_train_data_loader(data, labels, batch_size):
    return create_data_loader(data, labels, batch_size, shuffle=True)


def create_test_data_loader(data, labels, batch_size):
    return create_data_loader(data, labels, batch_size, shuffle=False)