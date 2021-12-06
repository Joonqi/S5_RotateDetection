# 합성 데이터를 이용한 문서의 회전 각도 예측 모델 학습

[🔗 Inference Example (Colab Notebook)](https://colab.research.google.com/drive/15iwinwQxKW4nObUGq3y1He-imueVBS3l#scrollTo=U8uvW6e396oS)

[Rotate Detection (w Lomin)](https://www.notion.so/Rotate-Detection-w-Lomin-4504dead35394bd7ad9397f719da9c04#9e068d5e2e9b4bbe9b0f4f5a67205be5)

회전된 문서 이미지 사진가 현재 몇 도 회전되어 있는지 측정하는 모델입니다.  

회전된 문서 이미지의 각도를 예측하여 보정하는 합성곱 신경망 모델을 설계했습니다. 훈련 데이터는 합성된 이미지로 데이터 증강기법들을 활용해 모델 학습을 진행했습니다.

기업 로민과의 협업으로 진행된 프로젝트입니다.

---

### 프로젝트 배경

- 문서를 데이터로 처리하는 데에 글자가 회전되어 있다면 인식 성능이 낮아질 수 있다.
이를 정방향으로 보정하는 과정은 모델 성능 향상에 도움이 된다.
→ 입력 문서 이미지의 회전 각도를 정확하게 예측하는 모델이 필요
- 딥러닝 모델을 학습시키기 위한 테이터는 실제로 수집하는 데에 많은 비용이 발생한다.
이를 최소화하기 위해 합성을 통해 인공적으로 데이터를 제작하고,
최대한 실제 데이터와 유사하게 증강시켜 효율적으로 학습을 진행한다.
→ 합성 데이터를 실제 데이터 분포와 유사하게 증강하여 실제 데이터에 대해서도 높은 성능을 보이는 모델 학습

---

### 프로젝트 세부 사항

- 각도를 예측하는 문제이기 때문에 회귀모델과 분류모델을 모두 고려했고 모델 성능의 이유로 360개의 클래스를 가진 분류모델로 학습
- Pytorch를 활용하여 Alexnet, VGG13, RepVGG 모델을 구현하고 성능 비교
- Torchvision의 증강기법을 조합하여 실제로 이미지가 어떤 형태를 띄는지, 모델 학습에는 어떤 영향을 미치는지 확인

### Model

```python
# RepVGG model from https://github.com/DingXiaoH/RepVGG

```

[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)

[GitHub - DingXiaoH/RepVGG: RepVGG: Making VGG-style ConvNets Great Again](https://github.com/DingXiaoH/RepVGG)


### Data Augmentation

```python
def get_train_trans(args):
    transform = tfs.Compose([
        tfs.CenterCrop(args.resolution),
        tfs.Grayscale(3),
        tfs.ColorJitter(.4, .4, 0, .4),
        tfs.ToTensor(),
        tfs.Normalize([.5, .5, .5], [.5, .5, .5]),        
        ])
    return transform
```
---

## Result

- 실제 문서 데이터인 테스트 이미지 4장을 각각 0 도~ 359 도로 회전시킨 총 1440개의 샘플로 정확도 측정
⇒ 정확도 34%, 오차 3도 이내 정확도 68%

![Error Cases](https://user-images.githubusercontent.com/83646259/144781003-fa92ea8e-861f-4bb3-b437-7c57f0350bf9.png)

Error Cases (1440 test samples)

- 100장의 실제 무작위로 회전된 문서 이미지에 적용해 본 결과 약 75장 정도 정확하게 보정
- 오류 케이스를 확인해보면 대부분 180도 오차를 보이고, 90도나 270도의 오차를 가진 경우도 존재
→ 뒤집힌 경우, 옆으로 누운 경우로 예측하는 경우가 많았다.
→ 문서 이미지를 Resize하고 Crop하는 과정에서 정보 손실이 발생해 문자 형태를 모델이 제대로 인식하지 못해 발생하는 문제로 추정
⇒ 향후 전처리 과정을 더 정교하게 설정하여 해결할 수 있을 것이라 기대

### Trained model Inference Example (Colab Notebook)

[Google Colaboratory](https://colab.research.google.com/drive/15iwinwQxKW4nObUGq3y1He-imueVBS3l#scrollTo=U8uvW6e396oS)

실제 문서 데이터인 테스트 이미지 4장을 각각 0 도~ 359 도로 회전시킨 총 1440개의 샘플로 정확도 측정
정확도 34%, 오차 3도 이내 정확도 68%

## 한계, 향후 개선 방향
- 전처리의 중요성 인식 :  
서로 다른 해상도의 사진이 들어올 때 더 효율적이고 정확한 자동화 리사이징 방식 적용 필요
- 모델 자체의 인식 성능 개선

### ++ OpenCV 를 활용하는 접근 방식

- OpenCV 라이브러리를 활용하여 딥러닝 모델을 활용하지 않고 비교적 간단하게 회전 각도를 예측할 수 있는 방법이 존재 (Skew Correction) :
텍스트 문단의 형태로 바운딩 박스를 작성하고 박스의 기울기를 측정하는 방식
- 실제로 해당 방식은 높은 정확도를 보이지만, +- 45도 범위 내에서만 정확하게 작동, 
360도 범위에서는 정확도를 기대하기 어려움
- 하지만 무작위로 회전된 이미지 데이터를 상하좌우 4개 클래스로 먼저 예측하고
바운딩 박스 기울기를 측정하는 두 단계를 거쳤을 때,
4개 클래스 분류의 정확도는 360 클래스 분류보다 훨씬 높은 성능이 나올 것이기 때문에 
결과적으로 더 높은 성능, 더 빠른 추정 결과가 나올 가능성이 높은 것으로 판단
- 추후에 실제로 구현을 시도해 볼 예정
