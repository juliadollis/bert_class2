import os
import json
import regex as re
import requests
import torch

def bytes_to_unicode():
    """
    Retorna um dicionário mapeando bytes para caracteres unicode.
    Este método é usado para garantir que todos os bytes sejam representáveis como caracteres únicos.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

def get_pairs(word):
    """
    Retorna um conjunto de pares adjacentes em uma palavra.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges):
        """
        Inicializa o codificador BPE.
        
        Args:
            encoder (dict): Dicionário mapeando tokens para IDs.
            bpe_merges (list of tuples): Lista de pares de tokens para asfusos BPE.
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder  # Dicionário de tokens para IDs
        self.decoder = {v: k for k, v in self.encoder.items()}  # Dicionário de IDs para tokens
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache = {}

        # Adiciona o token <pad> se não estiver presente
        if '<pad>' not in self.encoder:
            pad_id = len(self.encoder)
            self.encoder['<pad>'] = pad_id
            self.decoder[pad_id] = '<pad>'
            print(f"Adicionado token <pad> com ID {pad_id}")

    def bpe(self, token):
        """
        Aplica BPE (Byte Pair Encoding) em um token.
        
        Args:
            token (str): Token a ser codificado.
        
        Returns:
            str: Token codificado após asfusos BPE.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            # Encontra o par de maior prioridade (menor índice)
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        Codifica um texto em uma lista de índices de tokens.
        
        Args:
            text (str): Texto a ser codificado.
        
        Returns:
            list of int: Lista de índices de tokens codificados.
        """
        bpe_idx = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join([self.byte_encoder[b] for b in token_bytes])
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """
        Codifica um texto e mostra detalhadamente o processo.
        
        Args:
            text (str): Texto a ser codificado.
        
        Returns:
            dict: Dicionário contendo os índices de tokens e detalhes do processo.
        """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join([self.byte_encoder[b] for b in token_bytes])
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        out = {
            'bpe_idx': bpe_idx,
            'tokens': tokens,
            'parts': parts,
        }
        return out

    def decode(self, bpe_idx):
        """
        Decodifica uma lista de índices de tokens de volta para o texto original.
        
        Args:
            bpe_idx (list of int): Lista de índices de tokens.
        
        Returns:
            str: Texto decodificado.
        """
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

    def __len__(self):
        """
        Retorna o tamanho do vocabulário.
        
        Returns:
            int: Tamanho do vocabulário.
        """
        return len(self.encoder)

def get_file(local_file, remote_file):
    """
    Baixa um arquivo remoto se ele não existir localmente.
    
    Args:
        local_file (str): Caminho local para salvar o arquivo.
        remote_file (str): URL do arquivo remoto.
    """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        with open(local_file, "wb") as f:
            f.write(response.content)

def get_encoder():
    """
    Obtém o codificador BPE baixando os arquivos necessários se ainda não estiverem presentes.
    
    Returns:
        Encoder: Instância do codificador BPE.
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # Baixar encoder.json
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257, "Tamanho do encoder.json esperado é 50257"

    # Baixar vocab.bpe
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000, "Tamanho de bpe_merges esperado é 50000"

    # Instanciar o Encoder
    enc = Encoder(encoder, bpe_merges)
    return enc

class BPETokenizer:
    def __init__(self):
        """
        Inicializa o tokenizador BPE.
        """
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        """
        Codifica um texto em tensores PyTorch.
        
        Args:
            text (str): Texto a ser codificado.
            return_tensors (str): Tipo de tensor a ser retornado. Atualmente suporta apenas 'pt'.
        
        Returns:
            torch.Tensor: Tensor contendo os índices de tokens.
        """
        assert return_tensors == 'pt', "Atualmente, apenas 'pt' é suportado para return_tensors"
        assert isinstance(text, str), "O texto deve ser uma string"
        idx = self.encoder.encode(text)
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        """
        Decodifica uma lista de índices de tokens de volta para o texto original.
        
        Args:
            idx (torch.Tensor): Tensor contendo os índices de tokens.
        
        Returns:
            str: Texto decodificado.
        """
        assert idx.ndim == 1, "O tensor de índices deve ser unidimensional"
        text = self.encoder.decode(idx.tolist())
        return text

    def encode(self, text):
        """
        Codifica um texto em uma lista de índices de tokens.
        
        Args:
            text (str): Texto a ser codificado.
        
        Returns:
            list of int: Lista de índices de tokens codificados.
        """
        return self.encoder.encode(text)

if __name__ == '__main__':
    # Exemplo de uso do tokenizador
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    tokenizer = BPETokenizer()
    encoded = tokenizer.encode_and_show_work(text)
    
    print("Original text is:")
    print(text)
    print("\nFirst the text gets pre-tokenized:")
    print(encoded['tokens'])
    print("\nDetailed steps:")
    for part in encoded['parts']:
        print(part)
    print("\nFinal outcome (bpe_idx):")
    print(encoded['bpe_idx'])
    print("\nReady to feed into a Transformer!")
