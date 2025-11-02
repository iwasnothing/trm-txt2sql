# Fix TensorBoard Protobuf Warning

If you see the warning:
```
ImportError: cannot import name 'notf' from 'tensorboard.compat'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

This is due to protobuf version incompatibility. Fix it by:

```bash
pip install "protobuf>=3.20.0,<5.0.0"
```

Or if you're in a conda environment:
```bash
conda install "protobuf>=3.20.0,<5.0.0"
```

The training will still work, but fixing this will remove the warning.
