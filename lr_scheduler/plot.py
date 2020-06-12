import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=0)

x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    y.append(scheduler.get_lr()[0])

plt.plot(x, y)
plt.savefig('CosineAnnealingLR.jpg')
