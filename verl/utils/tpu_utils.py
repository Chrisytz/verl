import torch

class Tpu:
    """This is to simulate torch.tpu backend. 
    Please refer to each individual function in torch.cuda documentation to understand what it does.
    """
    def __init__(self, device_name):
        import torch_xla
        import torch_xla.core.xla_model as xm
        self.torch_xla = torch_xla
        self.xm = xm
        self.device_name = device_name
    
    def is_available(self):
        return True

    def current_device(self):
        return self.xm.xla_device()

    def set_device(self, local_rank):
        return
    
    def get_device_name(self):
        return ""
    
    def empty_cache(self):
        return
    
    def memory_allocated(self):
        return self.xm.get_memory_info()["peak_bytes_used"]

    def memory_reserved(self):
        return self.xm.get_memory_info()["bytes_used"]
    
    def mem_get_info(self):
        mem_info = self.xm.get_memory_info()
        return mem_info["bytes_limit"] - mem_info["bytes_used"], mem_info["bytes_limit"]

    def max_memory_allocated(self):
        return self.xm.get_memory_info()["peak_bytes_used"]

    def max_memory_reserved(self):
        return self.xm.get_memory_info()["bytes_used"]
    
    def get_rng_state(self):
        return self.xm.get_rng_state()
    
    def set_rng_state(self, seed):
        self.xm.set_rng_state(seed)

    def manual_seed(self, seed):
        self.torch_xla.manual_seed(seed)

def shard_input_data(batch_values):
    import torch_xla
    import torch_xla.distributed.spmd as xs
    for tensor in batch_values:
        if not torch_xla._XLAC._get_xla_sharding_spec(tensor):
            partition_spec = tuple("fsdp" if i == 0 else None for i in range(tensor.ndim))
            xs.mark_sharding(tensor, xs.get_global_mesh(), partition_spec)

def conditional_gpu_logger(strategy, role, logger):
    """Returns GPUMemoryLogger decorator if not running on TPU."""
    from verl.utils.debug import GPUMemoryLogger
    if strategy != "xla":
        return GPUMemoryLogger(role=role, logger=logger)
    else:
        def no_op_decorator(func):
            return func
        return no_op_decorator
