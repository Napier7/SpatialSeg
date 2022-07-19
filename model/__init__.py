from model import net, resnet, segformer, deeplabv3

def get_model(type):
    model = None
    if type == 'net':
        model = net.Net
    elif type == 'resnet':
        model = resnet.Resnet
    elif type == 'segformer':
        model = segformer.Segformer
    elif type == 'deeplabv3':
        model = deeplabv3.DeepLabV3
    # elif type == '...'
    #     write here...
    else:
        return ValueError
    return model