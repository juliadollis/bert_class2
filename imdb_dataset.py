import os
import torch
from torch.utils.data import Dataset
from bpe import BPETokenizer
import tarfile
import urllib.request

class IMDBDataset(Dataset):
    def __init__(self, split='train', tokenizer=None, max_length=512):
        """
        Args:
            split (str): 'train' ou 'test'
            tokenizer (BPETokenizer): Tokenizer para processar os textos
            max_length (int): Comprimento máximo dos tokens
        """
        assert split in ['train', 'test']
        self.split = split
        self.tokenizer = tokenizer or BPETokenizer()
        self.max_length = max_length
        self.data, self.labels = self._load_data()

    def _download_and_extract(self):
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        filename = 'aclImdb_v1.tar.gz'
        if not os.path.exists(filename):
            print("Baixando o conjunto de dados IMDb...")
            urllib.request.urlretrieve(url, filename)
        if not os.path.exists('aclImdb'):
            print("Extraindo o conjunto de dados IMDb...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall()
    
    def _load_data(self):
        self._download_and_extract()
        data = []
        labels = []
        path = os.path.join('aclImdb', self.split)
        for sentiment in ['pos', 'neg']:
            sentiment_path = os.path.join(path, sentiment)
            for fname in os.listdir(sentiment_path):
                if fname.endswith('.txt'):
                    with open(os.path.join(sentiment_path, fname), 'r', encoding='utf-8') as f:
                        text = f.read()
                        data.append(text)
                        labels.append(1 if sentiment == 'pos' else 0)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        # Tokenizar o texto
        tokens = self.tokenizer.encode(text)
        # Truncar ou padronizar o comprimento
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.tokenizer.encoder['<pad>']] * (self.max_length - len(tokens))
        # Criar máscara
        mask = [1] * min(len(self.tokenizer.encode(text)), self.max_length)
        mask += [0] * (self.max_length - len(mask))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
