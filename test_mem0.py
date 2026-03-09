import inspect
try:
    from mem0 import Memory
    print(inspect.signature(Memory.add))
except ImportError:
    pass
