import os, io
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import numpy as np
from vision_transformer.models import ViT
import random
import yaml
import argparse
from utils.log import TensorboardLogger
from dataset.dataset_getter import DatasetGetter

# Defining superclass and fine class names for CIFAR-100
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
        return "Unknown"  # Si el índice está fuera de rango, retorna "Unknown"
    
    superclass_index = class_index // 5
    fine_label_within_superclass = class_index % 5
    
    if superclass_index >= len(fine_class_names) or fine_label_within_superclass >= len(fine_class_names[superclass_index]):
        return "Unknown"  # Si los índices calculados están fuera de rango, retorna "Unknown"
    
    return fine_class_names[superclass_index][fine_label_within_superclass]

# Function to load a random image from CIFAR-100
def load_random_cifar100_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    image_index = random.randint(0, len(cifar_dataset) - 1)
    image, fine_label = cifar_dataset[image_index]
    superclass_index = fine_label // 5
    return image, fine_label, superclass_index  # Returns the image, fine label, and superclass index

# Function to perform inference with the model
def test_model_with_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        #print(predicted_label)
        predicted_class = get_class_name(predicted_label)
        #print(predicted_class)
        return outputs, predicted_label, predicted_class # Returns logits, predicted label, and predicted class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ViT model on CIFAR-100")
    
    # Arguments for device and model loading
    parser.add_argument(
        "--load-from",
        type=str,
        required=True,
        help="Path to the model checkpoint (including config.yaml)",
    )
    args = parser.parse_args()
    model_dir = os.path.dirname(args.load_from)
    
    # Debug prints to check args.load_from
    #print(f"Loading model from: {args.load_from}")
    #print(f"Model directory: {model_dir}")
    
    # Load configuration from config.yaml if available
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

    # Prepare the model
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

    # Load model checkpoint using io.BytesIO
    #print("Loading model checkpoint...")
    model_path = './checkpoints/240625_232446/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=args.device))

    # Load a random image and its class and label from CIFAR-100
    image_np, fine_label, superclass_index = load_random_cifar100_image()
    image_tensor = image_np.unsqueeze(0)  # Convert to a batch of size 1
    
    # Perform inference with the model
    outputs, predicted_label, predicted_class = test_model_with_image(model, image_tensor, args.device)
    
    # Display results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    # Transpose the image to (32, 32, 3) format
    plt.imshow(np.transpose(image_np, (1, 2, 0)))
    plt.axis('off')
    plt.title(f'CIFAR-100 Image: {fine_class_names[superclass_index][fine_label % 5]}')

    plt.subplot(1, 2, 2)
    # Plot all n_classes probabilities
    predicted_probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
    plt.barh(np.arange(len(predicted_probs)), predicted_probs)
    plt.yticks(np.arange(len(predicted_probs)), np.arange(len(predicted_probs)))  # Assuming index labels here
    plt.xlabel('Probability')
    plt.title(f'Prediction: {predicted_class}')

    plt.tight_layout()
    plt.show()

    print(f'Prediction: {predicted_class}, Fine label: {fine_class_names[superclass_index][fine_label % 5]}')
