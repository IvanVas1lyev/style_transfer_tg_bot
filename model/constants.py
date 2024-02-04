import torch
import torchvision.models as models


def download_vgg19_model():
    weights = models.VGG19_Weights.DEFAULT
    return models.vgg19(weights=weights).features.eval()


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

content_layers_default = ('conv_4',)
style_layers_default = ('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5')

vgg19 = download_vgg19_model()

log_msgs = [
    'Настройка параметров модели...',
    'Улучшаю результаты...',
    'Анализирую результаты обучения...',
    'Исследую новые подходы...',
    'Тюнинг модели в работе...',
    'Постепенно приближаюсь к идеальной модели...',
    'Оптимизация в действии...',
    'Обновление в процессе...'
 ]
