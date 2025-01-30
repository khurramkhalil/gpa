from config import Config


def get_dataset():
    config = Config()

    if config.dataset == 'agnews':
        from .agnewsdataset import NewsDataset
    elif config.dataset == '20newsgroups':
        from .newsgroups_20_dataset import NewsDataset
    elif config.dataset == 'imdb':
        from .imdbdataset import NewsDataset
    elif config.dataset == 'amazonreviews':
        from .amazonreviewdataset import NewsDataset
    elif config.dataset == 'sogounews':
        from .sogounewsdataset import NewsDataset

    return NewsDataset
