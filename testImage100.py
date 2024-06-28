import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import numpy as np
from vision_transformer.models import ViT
import random
import yaml
import argparse
import pickle
from PIL import Image
from dataset.dataset_getter import DatasetGetter

superclass_names = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
    'trees', 'vehicles 1', 'vehicles 2'
]
fine_class_names = [
    ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout'],
    ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
    ['bottles', 'bowls', 'cans', 'cups', 'plates'],
    ['apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'],
    ['clock', 'computer keyboard', 'lamp', 'telephone', 'television'],
    ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    ['crab', 'lobster', 'snail', 'spider', 'worm'],
    ['baby', 'boy', 'girl', 'man', 'woman'],
    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    ['maple', 'oak', 'palm', 'pine', 'willow'],
    ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
    ['lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
]

def get_class_name(class_index):
    if class_index < 0 or class_index >= 100:
        return "Unknown"    
    superclass_index = class_index // 5
    fine_label_within_superclass = class_index % 5
    if superclass_index >= len(fine_class_names) or fine_label_within_superclass >= len(fine_class_names[superclass_index]):
        return "Unknown"
    return fine_class_names[superclass_index][fine_label_within_superclass]

def load_random_cifar100_image():
    def load_cifar100_batch(file):
        with open(file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict

    # Load CIFAR-100 data
    data_dir = './data/cifar-100-python'
    test_data = load_cifar100_batch(os.path.join(data_dir, 'test'))
    test_images = test_data[b'data']
    test_labels = test_data[b'fine_labels']

    # Randomize test images
    indices = list(range(len(test_images)))
    random.shuffle(indices)

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # Normalización para CIFAR-100
    ])

    # Select a random image
    random_index = indices[0]
    image_data = test_images[random_index]
    fine_label = test_labels[random_index]

    # Reshape and transform the image
    image_data = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
    image = Image.fromarray(image_data)
    image = transform(image)

    return image, fine_label

def test_model_with_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)  # Añadir una dimensión para el batch
        outputs = model(image_tensor)
        predicted_probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(predicted_probs, 1)
        predicted_label = predicted.item()
        predicted_class = get_class_name(predicted_label)
        return predicted_probs, predicted_label, predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ViT model on CIFAR-100")
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
    
    # Cargar una imagen aleatoria del CIFAR-100
    image_tensor, fine_label = load_random_cifar100_image()
    
    # Probar el modelo con la imagen cargada
    outputs, predicted_label, predicted_class = test_model_with_image(model, image_tensor, args.device)
    
    predicted_probs = outputs.cpu().numpy().squeeze()
    p_prob = np.max(predicted_probs)
    p_class = np.argmax(predicted_probs)
    
    # Mostrar la imagen y la predicción
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(image_tensor.cpu().numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title(f'Fine Label: {get_class_name(fine_label)}, Prediction: {predicted_class} (Confidence: {p_prob:.2f})')
    plt.tight_layout()
    plt.show()
    
    print(f'Prediction: {predicted_class}, Fine label: {get_class_name(fine_label)} (Confidence: {p_prob:.2f})')