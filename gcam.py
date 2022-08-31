import numpy as np
import argparse
import torch
from PIL import Image
# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from dataset import get_transform, showim
from train import get_model

if __name__ == "__main__":  #  メインプログラム
    """ main program for gardcam
    usage :
        python gcam.py --imgs n1[,n2,....] --model MODEL [--pretrained]
        MODEL and --pretrained are used to make trained model file name
    """
    parser = argparse.ArgumentParser(description='gcam')
    parser.add_argument('--imgs', type=int, nargs='*', required=True,
                        default=None, help='image ids')
    parser.add_argument('--outd', type=str, default='result', help='Output directory')
    parser.add_argument('--model', type=str, default='VGG16', help='CNN model')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    args = parser.parse_args()
    print(args)

    imfs = []   # image file names
    labels = []
    for i in args.imgs:
        if i < 3358: # day1 number less than 3358
            day = '1'
            labels.append(0)
        else:        # day2 
            day = '2'
            labels.append(1)
        imfs.append(f"sampledata/day{day}/{i}-input.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    n_outputs = 2
    # args.pretrained is used to make fname
    model = get_model(args.model, n_outputs, False) 
    fname = args.outd + '/' + args.model+ ('_pre' if args.pretrained else '')
    print(f"Load {fname}_model.pt")
    model.load_state_dict(torch.load(fname+ "_model.pt"))
    model = model.to(device)
    # for Grad-CAM
    if args.model == 'VGG16':
        target_layer = model.features 
    elif args.model == 'ResNet50':
        target_layer = model.layer4[2]
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)
    model.eval()
    images = []
    msg = []
    tr2 = get_transform('test')  # for gradcam process
    tr3 = get_transform('view')  # for image viewing
    for i,imf in enumerate(imfs):
        img = Image.open(imf)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        torch_img = tr2(img).to(device)
        output = model(torch_img.unsqueeze(dim=0)) # forward
        output_index = torch.argmax(output)        # get the predicted class
        print(f"Ground Truth = {labels[i]}, Prediction={output_index.item()}")
        # gradcam
        gc_mask, gc_output = gradcam(torch_img.unsqueeze(dim=0))
        heatmap, gc_image = visualize_cam(gc_mask, torch_img)
        # gradcam++
        gcpp_mask, gcpp_output = gradcam_pp(torch_img.unsqueeze(dim=0))
        heatmap_pp, gcpp_image = visualize_cam(gcpp_mask, torch_img)
        # make list to show result
        images.append([tr3(img), gc_image, gcpp_image])
        msg.append([f"GT={labels[i]}", f"Pred={output_index.item()}",
                    f"Pred={output_index.item()}"])
            
    ll = len(images)
    ims = [ images[j][i]  for i in range(3) for j in range(ll)]
    msg = [ msg[j][i]   for i in range(3) for j in range(ll)]
    showim(ims, nx=ll, ny=3, msg=msg, fname=None) # show result
