import torch
import torchvision.models as models
from thop import profile
from ptflops import get_model_complexity_info
from torchstat import stat
import time
from TransHRNet_ddw import TransHRNet_L
import os

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

print("**********************************************************************************************")
print("test vgg16")
net = models.vgg16()
input = torch.randn(1, 3, 224, 224) # batch_size = 1
# inputs = torch.randn(1, 1, 48, 192, 192)
flops, params = profile(net, inputs =(input, ))
print("Flops: {} G".format(flops/1e9))
print("params: {} M".format(params/1e6))

#stat(model, (3, 224, 224))
flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False) # 默认batch_size =1
print("Flops: {}".format(flops))
print("Params: "+params)

time_start=time.time()
output = net(input)
time_end=time.time()
print('totally cost',time_end-time_start)

print("**********************************************************************************************")
print("\n"*2)
# stat(net, (3, 224, 224))

size = 192
print("test TransHRNet_L"+"\n")
our_network = TransHRNet_L(norm_cfg="IN", activation_cfg="LeakyReLU", img_size=[48,size, size],
                          num_classes=14, weight_std=False, deep_supervision=True).cuda()

flops, params = get_model_complexity_info(our_network, (1, 48, size, size), as_strings=True, print_per_layer_stat=False)
print("Flops: {}".format(flops))
print("Params: "+params)

input = torch.randn(1, 1, 48, size, size).cuda()
time_start=time.time()
output = our_network(input)
time_end=time.time()
print('totally cost',time_end-time_start)
print("\n"*3)