
datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(
    name,
    is_pretrain,
    # split=None, 
    # is_training=False, 
    **kwargs):
    """
        A simple dataset builder
    """
    if is_pretrain:
      dataset = datasets[name](**kwargs)
    else:
        dataset = datasets[name](**kwargs)
    return dataset