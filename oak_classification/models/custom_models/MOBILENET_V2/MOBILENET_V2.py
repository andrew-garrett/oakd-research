import torch.nn as nn
import torchvision.models

# class MOBILENET_V2:
# 	def __init__(self, pretrained=True, num_classes=10, feature_extracting=True):
def MOBILENET_V2(pretrained=True, num_classes=10, feature_extracting=True):
	# Create a model instance of the MobileNet V2 Architecture, pretrained on ImageNet
	model = torchvision.models.mobilenet_v2(pretrained=pretrained)
	# If specified, freeze the feature extractor
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False
	# Replace the classifier (which outputs 1000 classes for ImageNet) with a classifier which outputs 10 classes for CIFAR-10
	model.classifier[1] = nn.Linear(in_features=model.last_channel, out_features=num_classes)
	return model