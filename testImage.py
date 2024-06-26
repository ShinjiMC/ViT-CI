import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vision_transformer.models import ViT
import random

# Función para cargar una imagen aleatoria de CIFAR-10
def load_random_cifar10_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    image_index = random.randint(0, len(cifar10_dataset) - 1)
    image, label = cifar10_dataset[image_index]
    return image, label  # Devuelve la imagen y la etiqueta

# Función para realizar la inferencia con el modelo
def test_model_with_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return outputs, predicted.item()  # Devuelve los logits y la etiqueta predicha como un número entero

if __name__ == "__main__":
    # Configuración de dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar una imagen aleatoria de CIFAR-10
    image, true_label = load_random_cifar10_image()
    image_tensor = image.unsqueeze(0)  # Convertir a un batch de tamaño 1
    
    # Cargar el modelo ViT preentrenado
    model = ViT(
        image_size=32,
        n_channel=3,
        n_patch=16,
        n_dim=768,
        n_encoder_blocks=12,
        n_heads=12,
        n_classes=10,
        use_cnn_embedding=False,
    ).to(device)
    
    # Cargar los pesos del modelo entrenado
    model_path = './checkpoints2/240623_231231/model_200.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Realizar la inferencia con el modelo
    outputs, predicted_label = test_model_with_image(model, image_tensor, device)
    
    # Mostrar resultados
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Convertir la imagen tensorial a numpy y cambiar el orden de los canales
    image_np = image.numpy().transpose((1, 2, 0))
    
    # Crear la figura de Matplotlib
    plt.figure(figsize=(8, 4))
    
    # Mostrar la imagen
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.axis('off')
    plt.title(f'Imagen de CIFAR-10: {class_names[true_label]}')
    
    # Mostrar los resultados de la predicción
    plt.subplot(1, 2, 2)
    plt.barh(np.arange(len(class_names)), torch.softmax(outputs, dim=1).cpu().numpy().squeeze())
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel('Probabilidad')
    plt.title(f'Predicción: {class_names[predicted_label]}')
    
    plt.tight_layout()
    plt.show()
    
    print(f'Predicción: {class_names[predicted_label]}, Etiqueta verdadera: {class_names[true_label]}')
