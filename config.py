PATCH_LENGTH = 128      # Patch Length
PATCH_SIZE = 32       # Patch Size

PATCH_NUM_LAYERS = 9         # Number of layers in the encoder
CHAR_NUM_LAYERS = 3          # Number of layers in the decoder

NUM_EPOCHS = 32                 # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 5e-5            # Learning rate for the optimizer
PATCH_SAMPLING_BATCH_SIZE = 0   # Batch size for patch during training, 0 for full context
LOAD_FROM_CHECKPOINT = False     # Whether to load weights from a checkpoint
SHARE_WEIGHTS = False            # Whether to share weights between the encoder and decoder
