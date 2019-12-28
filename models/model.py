from torchvision import models
import torch
from torch import nn

class FCN(nn.Module):
	def __init__(self, n_classes=1):
		super(FCN, self).__init__()

		self.encoder = models.vgg16(pretrained=True).features

		self.deconv0 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) 
		self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # 256 x /32 x /32
		self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) 
		self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # 512 x /32 x /32
		self.classifier = nn.ConvTranspose2d(32, n_classes, kernel_size=1) 
		self.relu = torch.relu

	def forward(self, x):
		"""

		Forward Pass
		"""
		# Encoder
		x = self.encoder(x) # 512 x /32 x /32
		
		# Decoder
		x = self.relu(self.deconv0(x)) # 512 x /16 x /16
		x = self.relu(self.deconv1(x)) # 256 x /8 x /6
		x = self.relu(self.deconv2(x)) # 128 x /4 x /4
		x = self.relu(self.deconv3(x)) # 64 x /2 x /2
		x = self.relu(self.deconv4(x)) # 32 x /1 x /1
		x = self.classifier(x) # n_classes x /1 x /1

		x = torch.sigmoid(x)

		return x