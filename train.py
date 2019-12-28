from models.model import FCN
import numpy as np
import torch
from utils.utils import binarize_photo
import cv2

image = cv2.imread("face_image.jpeg")

image = torch.Tensor(image).view(1, 3, 512, 512)

label = cv2.imread("image_mask.jpeg")[:, :, 0] / 255.0

label = torch.Tensor(label).view(1, 1, 512, 512)

criterion = torch.nn.BCELoss()

net = FCN()

opt = torch.optim.Adam(net.parameters(), lr=1e-2)

for _ in range (10):

	model_out = net(image)
	loss = criterion(model_out, label)
	opt.zero_grad()

	loss.backward()

	print (loss)

	opt.step()
