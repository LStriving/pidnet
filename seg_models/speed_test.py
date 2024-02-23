import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import unet
import enet
import bisenet

def parse_args():
    parser = argparse.ArgumentParser(description='Speed Measurement')
    
    parser.add_argument('--a', help='unet or enet or bisenet', default='unet', type=str)
    parser.add_argument('--c', help='number of classes', type=int, default=11)
    parser.add_argument('--r', help='input resolution', type=int, nargs='+', default=(1080,1920))     


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    if args.a == 'unet':
        model = unet.UNet(args.c)
    elif args.a == 'enet':
        model = enet.ENet(args.c)
    elif args.a == 'bisenet':
        model = bisenet.BiSeNetv1(args.c)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, args.r[0], args.r[1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)