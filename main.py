import os
import argparse
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset.dataset_getter import DatasetGetter
from vision_transformer.models import ViT
from vision_transformer.learner import ViTLearner
from utils.torch import get_device, save_model, load_model
from utils.log import TensorboardLogger
from utils.config import save_yaml, load_from_yaml


def get_current_time() -> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time

def evaluate_model(model, dataset_loader, device):
    model.eval()
    confidences = []
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidences.extend(confidence.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    return confidences, predictions, ground_truths


def plot_confidences(confidences):
    plt.figure(figsize=(10, 5))
    plt.hist(confidences, bins=20, alpha=0.7, label='Confidences')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidences')
    plt.legend()
    plt.show()

def get_dataset_and_loader(args, device):
    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name, path=args.dataset_path, is_train=not args.test
    )
    dataset_loader = DatasetGetter.get_dataset_loader(
        dataset=dataset, batch_size=1 if args.test else args.batch_size
    )
    sampled_data = next(iter(dataset_loader))[0].to(device)
    return dataset_loader, sampled_data

def load_model_config(args):
    dir_path = os.path.dirname(args.load_from)
    config_file_path = os.path.join(dir_path, "config.yaml")
    config = load_from_yaml(config_file_path)
    args.patch_size = config["patch_size"]
    args.embedding_size = config["embedding_size"]
    args.encoder_blocks_num = config["encoder_blocks_num"]
    args.heads_num = config["heads_num"]
    args.classes_num = config["classes_num"]
    
def create_model(args, n_channel, image_size, device):
    return ViT(
        image_size=image_size,
        n_channel=n_channel,
        n_patch=args.patch_size,
        n_dim=args.embedding_size,
        n_encoder_blocks=args.encoder_blocks_num,
        n_heads=args.heads_num,
        n_classes=args.classes_num,
        use_cnn_embedding=args.use_cnn_embedding,
    ).to(device)
    
def setup_training(args, model, sampled_data):
    model_save_dir = os.path.join(args.save_dir, get_current_time())
    logger = TensorboardLogger(model_save_dir)
    logger.add_model_graph(model=model, image=sampled_data)
    save_yaml(vars(args), os.path.join(model_save_dir, "config.yaml"))
    return model_save_dir, logger

def run_epoch(learner, dataset_loader, is_train, device):
    loss_list, acc_list = [], []
    for images, labels in dataset_loader:
        images, labels = images.to(device), labels.to(device)
        loss, acc = learner.step(images=images, labels=labels, is_train=is_train)
        loss_list.append(loss)
        acc_list.append(acc)
    return np.mean(loss_list), np.mean(acc_list)

def train_or_test(args, epoch, learner, dataset_loader, model, model_save_dir=None, logger=None):
    best_acc = 0.0
    best_epoch = 0
    train_loss_history = []
    train_acc_history = []

    for epoch in range(epoch):
        loss_avg, acc_avg = run_epoch(learner, dataset_loader, not args.test, args.device)
        
        if acc_avg > best_acc:
            best_acc = acc_avg
            best_epoch = epoch + 1
            if model_save_dir:
                save_model(model, model_save_dir, "best_model")

        if logger:
            if (epoch + 1) % args.save_interval == 0:
                save_model(model, model_save_dir, "epoch_{}".format(epoch + 1))
            logger.log(tag="Training/Loss", value=loss_avg, step=epoch + 1)
            logger.log(tag="Training/Accuracy", value=acc_avg, step=epoch + 1)

        train_loss_history.append(loss_avg)
        train_acc_history.append(acc_avg)
        print("[Epoch {}] Loss : {} | Accuracy : {}".format(epoch + 1, loss_avg, acc_avg))

    print("Best Accuracy: {} at Epoch: {}".format(best_acc, best_epoch))
    return train_loss_history, train_acc_history, best_acc, best_epoch

def plot_training_history(train_loss_history, train_acc_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def evaluate_and_print_results(args, model, model_save_dir, device):
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    load_model(model, best_model_path)

    test_dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name, path=args.dataset_path, is_train=False
    )
    test_loader = DatasetGetter.get_dataset_loader(
        dataset=test_dataset, batch_size=args.batch_size
    )
    confidences, predictions, ground_truths = evaluate_model(model, test_loader, device)
    plot_confidences(confidences)

    for i in range(10):  # Print first 10 examples
        print(f"Prediction: {predictions[i]}, Ground Truth: {ground_truths[i]}, Confidence: {confidences[i]}")

    correct_confidences = [conf for conf, pred, gt in zip(confidences, predictions, ground_truths) if pred == gt]
    general_confidence = np.mean(correct_confidences)
    print("General Confidence of the Model: {:.2f}%".format(general_confidence * 100))

def get_current_time():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def run(args):
    device = get_device(args.device)
    dataset_loader, sampled_data = get_dataset_and_loader(args, device)
    n_channel, image_size = sampled_data.size()[1:3]
    if args.load_from and args.load_model_config:
        load_model_config(args)
    model = create_model(args, n_channel, image_size, device)
    if args.load_from is not None:
        load_model(model, args.load_from)
    learner = ViTLearner(model=model)
    epoch = 1 if args.test else args.epoch
    if not args.test:
        model_save_dir, logger = setup_training(args, model, sampled_data)
    train_loss_history, train_acc_history, _, _ = train_or_test(
        args, epoch, learner, dataset_loader, model, model_save_dir if not args.test else None, logger if not args.test else None
    )
    if not args.test:
        logger.close()
    plot_training_history(train_loss_history, train_acc_history)
    evaluate_and_print_results(args, model, model_save_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer")
    # dataset
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name to use GPU (ex. cpu, cuda, mps, etc.)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="cifar100", help="Dataset name (ex. cifar10"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="data/", help="Dataset path"
    )
    parser.add_argument("--classes-num", type=int, default=100, help="Number of classes")
    parser.add_argument("--patch-size", type=int, default=16, help="Image patch size")
    parser.add_argument(
        "--embedding-size", type=int, default=768, help="Number of hidden units"
    )
    parser.add_argument(
        "--encoder-blocks-num",
        type=int,
        default=12,
        help="Number of transformer encoder blocks",
    )
    parser.add_argument(
        "--heads-num", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--use-cnn-embedding", action="store_true", help="Whether to use cnn based patch embedding")
    # train / test
    parser.add_argument("--epoch", type=int, default=200, help="Learning epoch")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    # save / load
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/", help="Dataset name (ex. cifar10"
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="Model save interval"
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    parser.add_argument(
        "--load-model-config",
        action="store_true",
        help="Whether to use the config file of the model to be loaded",
    )

    args = parser.parse_args()
    run(args)
