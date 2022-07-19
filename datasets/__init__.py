from datasets import dataset

def get_dataset(type):
    if type == 'general':
        return dataset.GeneralDataset
    # elif type == '...'
    #     write here...
    else:
        return ValueError