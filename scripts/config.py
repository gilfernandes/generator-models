import torch
from pathlib import Path
import yaml


class Config():
    
    def __init__(self, config_file):
        self.name = config_file
        conf = yaml.safe_load(Path(config_file).read_text())
        self.batch_size = conf['batch_size']
        self.block_size = conf['block_size']
        self.max_iters = conf['max_iters']
        self.eval_interval = conf['eval_interval']
        self.learning_rate = float(conf['learning_rate'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = conf['eval_iters']
        self.n_embed = conf['n_embed']
        self.manual_seed = 1337
        self.n_layer = conf['n_layer']
        self.n_head = conf['n_head']
        self.dropout = conf['dropout']
        self.vocab_size = -1
        self.input_file = conf['input_file']
        self.generated_characters = conf['generated_characters']
        self.model_save_threshold = conf['model_save_threshold']
        
        self.data_folder = Path("../data")
        assert self.data_folder.exists()
        
