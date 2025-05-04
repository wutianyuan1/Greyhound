import abc
import dataclasses


@dataclasses.dataclass
class BaseConfig(abc.ABC):
    def to_config_string(self) -> str:
        config_str = ""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) and value:
                config_str += f"--{field.name.replace('_', '-')} "
            elif not isinstance(value, bool):
                config_str += f"--{field.name.replace('_', '-')} {value} "
        return config_str.strip()


@dataclasses.dataclass
class DistributedConfig(BaseConfig):
    nproc_per_node: int
    nnodes: int
    node_rank: int
    master_addr: str
    master_port: str


@dataclasses.dataclass
class DatasetConfig(BaseConfig):
    vocab_file: str
    merge_file: str
    # data_path: str
    split: str
    mock_data: bool


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int
    max_position_embeddings: int
    micro_batch_size: int
    global_batch_size: int
    lr: float
    train_iters: int
    lr_decay_iters: int
    lr_decay_style: str
    min_lr: float
    weight_decay: float
    lr_warmup_fraction: str
    clip_grad: float
    fp16: bool
    loss_scale: float
    failslow_aware: bool


@dataclasses.dataclass
class TrainConfig(BaseConfig):
    distributed_config: DistributedConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig
    log_interval: int
    save_interval: int
    eval_interval: int
    eval_iters: int
    distributed_backend: str
    save: str   # checkpoint save path
    load: str   # checkpoint load path

    def to_config_string(self) -> str:
        other_config_str = ""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, BaseConfig):
                other_config_str += f"--{field.name.replace('_', '-')} {value} "
        return "torchrun {} pretrain_gpt.py {} {} --sequence-parallel {}".format(
            self.distributed_config.to_config_string(),
            self.model_config.to_config_string(),
            self.dataset_config.to_config_string(),
            other_config_str
        ).strip()
