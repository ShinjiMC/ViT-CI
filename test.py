import torch

def convert_checkpoint_to_pth(checkpoint_path, save_path):
    # Cargar el archivo de checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    
    # Crear un diccionario para el estado del modelo
    model_state_dict = {
        'patch_embedder.class_token': checkpoint['patch_embedder.class_token'],
        'patch_embedder.position_embedding': checkpoint['patch_embedder.position_embedding'],
        'patch_embedder.projection.weight': checkpoint['patch_embedder.projection.weight'],
        'patch_embedder.projection.bias': checkpoint['patch_embedder.projection.bias']
    }
    
    for i in range(12):  # Iterar sobre los encoders (asumiendo 12 encoders en tu modelo)
        encoder_prefix = f'encoders.{i}.'
        model_state_dict.update({
            encoder_prefix + 'layer_norm.weight': checkpoint[encoder_prefix + 'layer_norm.weight'],
            encoder_prefix + 'layer_norm.bias': checkpoint[encoder_prefix + 'layer_norm.bias'],
            encoder_prefix + 'multi_head_attention.query.weight': checkpoint[encoder_prefix + 'multi_head_attention.query.weight'],
            encoder_prefix + 'multi_head_attention.query.bias': checkpoint[encoder_prefix + 'multi_head_attention.query.bias'],
            encoder_prefix + 'multi_head_attention.key.weight': checkpoint[encoder_prefix + 'multi_head_attention.key.weight'],
            encoder_prefix + 'multi_head_attention.key.bias': checkpoint[encoder_prefix + 'multi_head_attention.key.bias'],
            encoder_prefix + 'multi_head_attention.value.weight': checkpoint[encoder_prefix + 'multi_head_attention.value.weight'],
            encoder_prefix + 'multi_head_attention.value.bias': checkpoint[encoder_prefix + 'multi_head_attention.value.bias'],
            encoder_prefix + 'multi_head_attention.linear.weight': checkpoint[encoder_prefix + 'multi_head_attention.linear.weight'],
            encoder_prefix + 'multi_head_attention.linear.bias': checkpoint[encoder_prefix + 'multi_head_attention.linear.bias'],
            encoder_prefix + 'mlp_block.linear1.weight': checkpoint[encoder_prefix + 'mlp_block.linear1.weight'],
            encoder_prefix + 'mlp_block.linear1.bias': checkpoint[encoder_prefix + 'mlp_block.linear1.bias'],
            encoder_prefix + 'mlp_block.linear2.weight': checkpoint[encoder_prefix + 'mlp_block.linear2.weight'],
            encoder_prefix + 'mlp_block.linear2.bias': checkpoint[encoder_prefix + 'mlp_block.linear2.bias']
        })
    
    # Agregar la capa clasificadora
    model_state_dict.update({
        'classifier.linear.weight': checkpoint['classifier.linear.weight'],
        'classifier.linear.bias': checkpoint['classifier.linear.bias']
    })
    
    # Guardar solo el estado del modelo en un archivo .pth
    torch.save(model_state_dict, save_path)
    print(f"Modelo guardado como {save_path}")

if __name__ == "__main__":
    checkpoint_path = r'.\checkpoints2\240623_231231\epoch_200'  # Ruta al archivo epoch_200
    save_path = r'.\checkpoints2\240623_231231\model_200.pth'  # Ruta para guardar el archivo .pth
    
    convert_checkpoint_to_pth(checkpoint_path, save_path)
