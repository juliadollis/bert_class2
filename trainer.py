import time
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN

class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        C.device = 'auto'
        C.num_workers = 4
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        C.log_interval = 100  # Intervalo para logs
        C.save_interval = 1000  # Intervalo para salvar o modelo
        C.num_classes = 2  # Número de classes para classificação
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        # Métricas
        self.train_loss = 0.0
        self.train_correct = 0
        self.train_total = 0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def evaluate(self):
        if self.val_dataset is None:
            return
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_loader = DataLoader(
                self.val_dataset,
                sampler=torch.utils.data.SequentialSampler(self.val_dataset),
                shuffle=False,
                pin_memory=True,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
            )
            for batch in val_loader:
                x, labels, mask = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                logits, _ = self.model(x, mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc * 100:.2f}%")
        self.model.train()

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizers(config)
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, labels, mask = batch
            logits, lm_loss = model(x, mask)
            # Calcular a perda de classificação
            classification_loss = F.cross_entropy(logits, labels)
            # Combinar com a perda de modelagem de linguagem, se aplicável
            if lm_loss is not None:
                loss = classification_loss + lm_loss
            else:
                loss = classification_loss
            self.loss = loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            self.trigger_callbacks('on_batch_end')

            # Atualizar métricas
            self.train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            self.train_correct += (preds == labels).sum().item()
            self.train_total += labels.size(0)

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # Logs periódicos
            if self.iter_num % config.log_interval == 0:
                avg_loss = self.train_loss / config.log_interval
                accuracy = self.train_correct / self.train_total
                print(f"Iter {self.iter_num}: Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%")
                # Reset métricas
                self.train_loss = 0.0
                self.train_correct = 0
                self.train_total = 0

            # Avaliação periódica
            if self.iter_num % config.save_interval == 0:
                self.evaluate()

            # Checar condição de parada
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
