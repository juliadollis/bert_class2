class BERT(nn.Module):
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'bert'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.num_classes = 2  # Adicionado para classificação
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given  # Certifica que apenas um está definido
        if type_given:
            config.merge_from_dict({
                'gpt-mini':   dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':  dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':   dict(n_layer=3, n_head=3, n_embd=48),
            }.get(config.model_type, {}))

        # Resto da inicialização...
