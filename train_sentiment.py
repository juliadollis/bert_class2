import argparse
from bpe import BPETokenizer
from model import BERT
from trainer import Trainer
from imdb_dataset import IMDBDataset
from utils import CfgNode as CN, set_seed

def main():
    # Configuração de Argumentos de Linha de Comando
    parser = argparse.ArgumentParser(description="Treinamento de Classificação de Sentimentos com BERT")
    parser.add_argument('--config', type=str, default=None, help='Caminho para o arquivo de configuração JSON')
    parser.add_argument('--max_iters', type=int, default=10000, help='Número máximo de iterações de treinamento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Taxa de aprendizado')
    parser.add_argument('--seed', type=int, default=42, help='Semente para reprodutibilidade')
    args = parser.parse_args()

    # Inicializar Configuração
    config = Trainer.get_default_config()
    if args.config:
        # Carregar configurações adicionais de um arquivo JSON
        import json
        with open(args.config, 'r') as f:
            config_json = json.load(f)
        config.merge_from_dict(config_json)
    # Sobrescrever com argumentos de linha de comando, se fornecidos
    config.merge_from_dict({
        'max_iters': args.max_iters,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    })
    config.num_classes = 2  # Para classificação binária

    # Definir Semente
    set_seed(args.seed)

    # Inicializar Tokenizer
    tokenizer = BPETokenizer()

    # Carregar Datasets
    train_dataset = IMDBDataset(split='train', tokenizer=tokenizer, max_length=256)
    val_dataset = IMDBDataset(split='test', tokenizer=tokenizer, max_length=256)

    # Configurar Modelo
    model_config = BERT.get_default_config()
    model_config.n_layer = 6
    model_config.n_head = 6
    model_config.n_embd = 192
    model_config.vocab_size = len(tokenizer.encoder)
    model_config.block_size = 256
    model_config.num_classes = 2  # Número de classes para classificação
    model = BERT(model_config)

    # Inicializar Trainer
    trainer = Trainer(config, model, train_dataset, val_dataset)

    # Iniciar Treinamento
    trainer.run()

if __name__ == '__main__':
    main()
