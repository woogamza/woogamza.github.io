---
title: 파이토치 튜토리얼 1 - 파이토치의 기본기를 배워보자
author: 우감자
date: 2021-12-12 14:00:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, tutorial]
math: false
mermaid: false
image:
  src: /assets/img/posts/pytorch/tutorial/0/pytorch.png
  width: 850
  height: 585
---

# 파이토치 튜토리얼 1 - 파이토치의 기본기를 배우자

> To 딥러닝 이론을 배워도 쓸 수 없는 사람들에게...
> To 실제 프로그램에서 동작할 수 있는 코드를 작성해보려는 사람들에게..
> ref : 파이토치 한국어 튜토리얼

# 파이토치란?

## 개요

토치(Torch) 및 카페2(Caffe2)를 기반으로 한 텐서플로우와 유사한 딥러닝 라이브러리이다. 페이스북 인공지능 연구팀에 의해 주로 개발되어 왔다. 텐서플로우 2.0 이후의 친 keras 행보로 자유로운 네트워크 수정의 난이도가 점점 높아져, 최근 연구원들 사이에서는 PyTorch의 사용 비중이 높아지고 있다.(by 나무위키)

# 텐서(Tensor)

텐서는 배열(array)나 행렬(matrix)와 매우 유사한 자료구조입니다. 우리는 이 텐서를 모델의 입력부터 출력까지 행렬 연산에 지속적으로 사용하게 될 겁니다.
텐서는 GPU에서 실행할 수 있다는 점에서 Numpy의 ndarray와 차이점이 있습니다.
텐서는 Autograd(자동 미분)에 최적화되어 있습니다.

```
import torch
import numpy as np
```

## 텐서(tensor) 생성하기

**데이터로 직접 생성하기**

```
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

`torch.tensor(배열)`
이런 식으로 사용하면 직접 텐서를 생성할 수 있습니다.
**NumPy 배열로부터 생성하기**
텐서는 NumPy 배열로 생성가능합니다.

```
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**다른 텐서로부터 생성하기**
override하지 않는다면, 인자로 주어진 텐서의 속성(모양(shape), 자료형(datatype))을 유지합니다.

```
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n  {x_ones}  \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n  {x_rand}  \n")
```

**무작위(random) 또는 상수(constant)값을 사용하기**
`shape`은 텐서의 차원(dimension)값을 담고 있는 튜플(tuple)로, 아래 함수들에서 출력되는 텐서의 차원을 결정해줍니다.

```
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n  {rand_tensor}  \n")
print(f"Ones Tensor: \n  {ones_tensor}  \n")
print(f"Zeros Tensor: \n  {zeros_tensor}")
```

## 텐서의 속성(Attribute) 파악하기

텐서의 속성은 텐서의 `shape`, `datatype` 및 어느 장치에 저장되는지를 알려줍니다.

```
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

## 텐서 연산(Operation)

전치(transposing), 인덱싱(indexing), 슬라이싱(slicing), 수학 계산, 선형 대수, 임의 샘플링(random sampling) 등, 100가지 이상의 텐서 연산들을 확인할 수 있습니다. [link](https://pytorch.org/docs/stable/torch.html)
각 연산들은 GPU에서 실행할 수 있습니다. 따라서 우리는 컴퓨터에 GPU가 있는지 확인해볼 필요가 있습니다.

```
# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
```

`tensor.to('cuda')`를 통해 우리는 텐서를 GPU를 통해 연산할 수 있습니다.
**NumPy식 연산**
tensor를 다룰 때는 NumPy API와 거의 동일합니다.

```
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

**텐서 합치기**
`torch.cat`을 사용하면 텐서를 연결할 수 있습니다.

```
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

**산술 연산(Arithmetic operations)**

```
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

**single-element tensor**
텐서의 모든 값을 하나로 aggregate(집계)한 뒤, `item()`을 사용해 Python 숫자형 값으로 변환할 수 있습니다.

```
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

**in-place 연산**
in-place연산은 연산 후에 입력한 인자가 연산 결과로 바뀌는 연산입니다. `_`접미사를 갖습니다.
예를 들어 `x.copy_(y)`나 `x._()`는 연산 후에 `x`가 변경됩니다.

```
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

## Numpy 변환(Bridge)

tensor와 numpy 상호간에 변환을 했을 경우 CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경됩니다.

```
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

tensor를 통해 넘파이 변수 n을 선언했습니다.

```
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

t와 n변수 모두 값이 바뀌었음을 확인할 수 있습니다.

## NumPy 배열을 텐서로 변환해보기

```
n = np.ones(5)
t = torch.from_numpy(n)
```

`torch.from_numpy()`를 통해 numpy배열을 텐서로 변환할 수 있습니다.

```
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

NumPy 배열을 변경해도 텐서에 반영됩니다.

# DataLoader 만들기

딥러닝을 할 때 가장 애먹는 부분이라고도 말할 수 있습니다. 데이터 샘플을 처리하는 코드는 지저분해질 수 있기에 모델 학습 코드와 분리하는게 일반적입니다.
Pytorch에서는 `torch.utils.data.DataLoader`와 `torch.utils.data.Dataset` 두 가지 데이터 처리 api를 제공하기에 쉽게 데이터셋을 로드하고 데이터를 사용할 수 있습니다.
`DataLoader`는 Label값을 갖고 있는 데이터셋을 iterable한 객체로 감싸주는 역할을 합니다.
우리는 PyTorch의 위 2가지 api를 사용해 Fashion-MNIST데이터셋을 다뤄보도록 하겠습니다.

## 데이터셋 불러오기

TorchVision 에서 [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) 데이터셋을 불러오는 예제를 살펴보겠습니다. Fashion-MNIST는 Zalando의 기사 이미지 데이터셋으로 60,000개의 학습 예제와 10,000개의 테스트 예제로 이루어져 있습니다. 각 예제는 흑백(grayscale)의 28x28 이미지와 10개 분류(class) 중 하나인 정답(label)으로 구성됩니다.
다음 매개변수들을 사용하여 [FashionMNIST 데이터셋](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) 을 불러옵니다:

- `root` 는 학습/테스트 데이터가 저장되는 경로입니다.
- `train` 은 학습용 또는 테스트용 데이터셋 여부를 지정합니다.
- `download=True` 는 `root` 에 데이터가 없는 경우 인터넷에서 다운로드합니다.
- `transform` 과 `target_transform` 은 특징(feature)과 정답(label) 변형(transform)을 지정합니다.

```
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

## 데이터셋을 순회하고 시각화하기

`Dataset` 에 리스트(list)처럼 직접 접근(index)할 수 있습니다: `training_data[index]`. `matplotlib` 을 사용하여 학습 데이터의 일부를 시각화해보겠습니다.

```
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

## 파일에서 사용자 정의 데이터셋 만들기

사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다: `__init__`, `__len__`, `__getitem__`. 아래 구현을 살펴보면 FashionMNIST 이미지들은 `img_dir` 디렉토리에 저장되고, 정답은 `annotations_file` csv 파일에 별도로 저장됩니다.

```
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

## **init**

`__init__` 함수는 Dataset 객체가 생성(instantiate)될 때 한 번만 실행됩니다. 여기서는 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와 (다음 장에서 자세히 살펴볼) 두가지 변형(transform)을 초기화합니다.

labels.csv 파일은 다음과 같습니다:

tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9

```
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

## **len**

`__len__` 함수는 데이터셋의 샘플 개수를 반환합니다.

예:

```
def __len__(self):
    return len(self.img_labels)
```

## **getitem**

`__getitem__` 함수는 주어진 인덱스 `idx` 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다. 인덱스를 기반으로, 디스크에서 이미지의 위치를 식별하고, `read_image` 를 사용하여 이미지를 텐서로 변환하고, `self.img_labels` 의 csv 데이터로부터 해당하는 정답(label)을 가져오고, (해당하는 경우) 변형(transform) 함수들을 호출한 뒤, 텐서 이미지와 라벨을 Python 사전(dict)형으로 반환합니다.

```
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    sample = {"image": image, "label": label}
    return sample
```

---

# DataLoader로 학습용 데이터 준비하기

`Dataset` 은 데이터셋의 특징(feature)을 가져오고 하나의 샘플에 정답(label)을 지정하는 일을 한 번에 합니다. 모델을 학습할 때, 일반적으로 샘플들을 “미니배치(minibatch)”로 전달하고, 매 에폭(epoch)마다 데이터를 다시 섞어서 과적합(overfit)을 막고, Python의 `multiprocessing` 을 사용하여 데이터 검색 속도를 높이려고 합니다.

`DataLoader` 는 간단한 API로 이러한 복잡한 과정들을 추상화한 순회 가능한 객체(iterable)입니다.

```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## DataLoader를 통해 순회하기(iterate)

`DataLoader` 에 데이터셋을 불러온 뒤에는 필요에 따라 데이터셋을 순회(iterate)할 수 있습니다. 아래의 각 순회(iteration)는 (각각 `batch_size=64` 의 특징(feature)과 정답(label)을 포함하는) `train_features` 와 `train_labels` 의 묶음(batch)을 반환합니다. `shuffle=True` 로 지정했으므로, 모든 배치를 순회한 뒤 데이터가 섞입니다. (데이터 불러오기 순서를 보다 세밀하게(finer-grained) 제어하려면 [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler) 를 살펴보세요.)

## 이미지와 정답(label)을 표시합니다.

```
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

![../../_images/sphx_glr_data_tutorial_002.png](https://tutorials.pytorch.kr/_images/sphx_glr_data_tutorial_002.png)
