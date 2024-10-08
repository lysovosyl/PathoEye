import torch.nn as nn
from torchvision import models

class GradCamModel(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.pretrained = models.resnet50(pretrained=True)
        resnet50 = models.resnet50(pretrained=True)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, class_num), nn.LogSoftmax(dim=1))
        self.pretrained = resnet50
        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))
        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out