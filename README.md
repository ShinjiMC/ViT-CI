# SimpleViT

Simple implementation of Vision Transformer for Image Classification.

- DRL framework : PyTorch

## Install

```bash
git clone https://github.com/isk03276/SimpleViT
cd SimpleViT
pip install -r requirements.txt
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu111
pip install pyyaml six
pip install matplotlib
pip install tensorboard

```

## Getting Started

```bash
python main.py --dataset-name DATASET_NAME(ex. cifar10) --device DEVICE(ex. cuda, cpu) #train
python main.py --dataset-name DATASET_NAME(ex. cifar10) --device DEVICE(ex. cuda, cpu) --load-from MODEL_PATH --load-model-config #test
python main.py --dataset-name cifar10 --classes-num 10 --device cuda --epoch 200


python main.py --dataset-name cifar100 --classes-num 100 --device cuda --epoch 200

python main.py --dataset-name cifar100 --classes-num 100 --device cuda --epoch 200 --batch-size 128 --patch-size 16 --embedding-size 1024 --encoder-blocks-num 16 --heads-num 16
python main.py --dataset-name cifar100 --classes-num 100 --device cuda --epoch 200 --batch-size 128 --patch-size 16 --embedding-size 768 --encoder-blocks-num 12 --heads-num 12

python testImage.py

python testImage100.py --load-from ./checkpoints/240625_232446/best_model.pth

```

## Results

**- CIFAR-10**  
<img src="https://user-images.githubusercontent.com/23740495/190575759-317fe1fc-57a0-4771-abb1-41925d72e051.png" width="70%" height="70%"/>

**- CIFAR-100**  
<img src="https://user-images.githubusercontent.com/23740495/191168744-1871189b-5f7c-47e6-a005-35cc472457cf.png" width="70%" height="70%"/>
