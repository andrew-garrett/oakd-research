#################################################
#################### IMPORTS ####################
#################################################

import torch.nn as nn
from torchvision.models import mobilenet_v2

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
		self.model = mobilenet_v2(weights=weights)
		# If specified, freeze the feature extractor
		if feature_extracting:
			for param in self.model.parameters():
				param.requires_grad = False
		# Replace the classifier (which outputs 1000 classes for ImageNet) with a classifier which outputs 10 classes for CIFAR-10
		custom_classifier = nn.Sequential(
			nn.Linear(in_features=self.m.last_channel, out_features=4*num_classes),
			nn.Linear(in_features=4*num_classes, out_features=num_classes)
		)
		self.model.classifier[1] = custom_classifier

	def forward(self, x):
		return self.model(x)

if __name__ == "__main__":
	MOBILENET_V2()