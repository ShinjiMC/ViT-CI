import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from vision_transformer.models import ViT
import random
import yaml
import argparse
from PIL import Image
from dataset.dataset_getter import DatasetGetter

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_random_cifar10_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalización para CIFAR-10
    ])
    cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    image_index = random.randint(0, len(cifar10_dataset) - 1)
    image, label = cifar10_dataset[image_index]
    return image, label  # Devuelve la imagen y la etiqueta

def test_model_with_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        # Asegurarse de que la imagen tiene la forma correcta antes de pasar al modelo
        assert image_tensor.dim() == 4, f"Expected 4-dimensional input tensor, got {image_tensor.shape}"
        
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        predicted_probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(predicted_probs, 1)
        predicted_label = predicted.item()
        predicted_class = class_names[predicted_label]
        return predicted_probs, predicted_label, predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ViT model on CIFAR-10")
    parser.add_argument(
        "--load-from",
        type=str,
        required=True,
        help="Path to the model checkpoint (including config.yaml)",
    )
    args = parser.parse_args()
    model_dir = os.path.dirname(args.load_from)
    config_file_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        args.batch_size = config["batch_size"]
        args.classes_num = config["classes_num"]
        args.dataset_path = config["dataset_path"]
        args.dataset_name = config["dataset_name"]
        args.device = config["device"]
        args.embedding_size = config["embedding_size"]
        args.encoder_blocks_num = config["encoder_blocks_num"]
        args.epoch = config["epoch"]
        args.heads_num = config["heads_num"]
        args.load_from = config["load_from"]
        args.load_model_config = config["load_model_config"]
        args.patch_size = config["patch_size"]
        args.save_dir = config["save_dir"]
        args.save_interval = config["save_interval"]
        args.test = config["test"]
        args.use_cnn_embedding = config["use_cnn_embedding"]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name, path=args.dataset_path, is_train=not args.test
    )
    dataset_loader = DatasetGetter.get_dataset_loader(
        dataset=dataset, batch_size=1 if args.test else args.batch_size
    )
    sampled_data = next(iter(dataset_loader))[0].to(args.device)
    n_channel, image_size = sampled_data.size()[1:3]
    model = ViT(
        image_size=image_size,
        n_channel=n_channel,
        n_patch=args.patch_size,
        n_dim=args.embedding_size,
        n_encoder_blocks=args.encoder_blocks_num,
        n_heads=args.heads_num,
        n_classes=args.classes_num,
        use_cnn_embedding=args.use_cnn_embedding,
    ).to(args.device)
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    
    # Cargar una imagen aleatoria del CIFAR-10
    image, true_label = load_random_cifar10_image()
    image_tensor = image.unsqueeze(0)  # Convertir a un batch de tamaño 1
    
    # Probar el modelo con la imagen cargada
    outputs, predicted_label, predicted_class = test_model_with_image(model, image_tensor, args.device)
    
    predicted_probs = outputs.cpu().numpy().squeeze()
    p_prob = np.max(predicted_probs)
    p_class = np.argmax(predicted_probs)
    
    # Convertir la imagen tensorial a numpy y cambiar el orden de los canales
    image_np = image.numpy().transpose((1, 2, 0))
    
    # Mostrar la imagen y la predicción
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
    plt.title(f'Predicción: {class_names[predicted_label]} (Confidence: {p_prob:.2f})')
    
    plt.tight_layout()
    plt.show()
    
    print(f'Predicción: {class_names[predicted_label]}, Etiqueta verdadera: {class_names[true_label]}')
