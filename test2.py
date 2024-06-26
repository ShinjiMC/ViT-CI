import torch

checkpoint_path = r'.\checkpoints2\240623_231231\epoch_200'  # Ruta al archivo epoch_200

# Cargar el checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

# Imprimir las claves disponibles en el diccionario del checkpoint
print(checkpoint.keys())