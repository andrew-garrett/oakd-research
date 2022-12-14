#################################################
#################### IMPORTS ####################
#################################################

import torch
import torchvision
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


############################################################
#################### CUSTOM MODEL CLASS ####################
############################################################

class MOBILENET_V2(nn.Module):
	def __init__(self, pretrained=True, num_classes=10, feature_extracting=False):
		super().__init__()
		# Create a model instance of the MobileNet V2 Architecture, pretrained on ImageNet
		if pretrained:
			weights = "DEFAULT"
		else:
			weights = None
		self.m = torchvision.models.mobilenet_v2(weights=weights)
		# If specified, freeze the feature extractor
		if feature_extracting:
			for param in self.m.parameters():
				param.requires_grad = False
		# Replace the classifier (which outputs 1000 classes for ImageNet) with a classifier which outputs 10 classes for CIFAR-10
		custom_classifier = nn.Sequential(
			nn.Linear(in_features=self.m.last_channel, out_features=4*num_classes),
			nn.Linear(in_features=4*num_classes, out_features=num_classes)
		)
		self.m.classifier[1] = custom_classifier

	def forward(self, x):
		return self.m(x)



# def MOBILENET_V2(pretrained=True, num_classes=10, feature_extracting=False):
# 	# Create a model instance of the MobileNet V2 Architecture, pretrained on ImageNet
# 	if pretrained:
# 		weights = "DEFAULT"
# 	else:
# 		weights = None
# 	model = torchvision.models.mobilenet_v2(weights=weights)
# 	# If specified, freeze the feature extractor
# 	if feature_extracting:
# 		for param in model.parameters():
# 			param.requires_grad = False
# 	# Replace the classifier (which outputs 1000 classes for ImageNet) with a classifier which outputs 10 classes for CIFAR-10
# 	model.classifier[1] = nn.Linear(in_features=model.last_channel, out_features=num_classes)
# 	return model