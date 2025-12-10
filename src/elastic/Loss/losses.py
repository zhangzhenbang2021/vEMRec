import numpy as np
import torch


def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]

    det = (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) - \
          (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])

    return det


def elem_sym_polys_of_eigen_values(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]

    sigma1 = (M[0][0] + M[1][1] + M[2][2])

    sigma2 = (M[0][0] * M[1][1] + M[1][1] * M[2][2] + M[2][2] * M[0][0]) - \
             (M[0][1] * M[1][0] + M[1][2] * M[2][1] + M[2][0] * M[0][2])

    sigma3 = (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) - \
             (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])

    return sigma1, sigma2, sigma3


def similarity_loss(img1, img2_warped):
    sizes = np.prod(img1.shape[1:])
    flatten1 = img1.view(-1, sizes)
    flatten2 = img2_warped.view(-1, sizes)

    mean1 = torch.mean(flatten1, -1).view(-1, 1)
    mean2 = torch.mean(flatten2, -1).view(-1, 1)
    var1 = torch.mean((flatten1 - mean1) ** 2, -1)
    var2 = torch.mean((flatten2 - mean2) ** 2, -1)

    conv12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), -1)
    pearson_r = conv12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

    raw_loss = 1 - pearson_r
    raw_loss = torch.sum(raw_loss)

    return raw_loss

# @torch.no_grad()
def cnn_encoder(img, model):
    '''
    img:(B,1,H,W)
    '''
    model.eval()
    _,feature = model(img)
    return feature


def encoder_loss(imgs1,img2,model,weights):
    '''
    imgs1:(B,1,3,H,W)  img2:(B,1,H,W)
    '''
    def L2Loss(y,yhead):
        return torch.sqrt(torch.sum((y-yhead)**2))
        # return torch.sum((y-yhead)**2)
    
    img1_channel1 = imgs1[:, :, 0, :, :]
    img1_channel2 = imgs1[:, :, 1, :, :]
    img1_channel3 = imgs1[:, :, 2, :, :] # (B,1,H,W)
    img1_encoder1 = cnn_encoder(img1_channel1,model)
    img1_encoder2 = cnn_encoder(img1_channel2,model)
    img1_encoder3 = cnn_encoder(img1_channel3,model)
    img2_encoder = cnn_encoder(img2,model)

    
    sim_loss = 0.0
    sim_loss = sim_loss + weights[0]*L2Loss(img1_encoder1, img2_encoder)
    sim_loss = sim_loss + weights[1]*L2Loss(img1_encoder2, img2_encoder)
    sim_loss = sim_loss + weights[2]*L2Loss(img1_encoder3, img2_encoder)
    
    return sim_loss


def regularize_loss2(flow):
    ret = torch.sum((flow[:, :, 1:, :] - flow[:, :, :-1, :]) ** 2) / 2 + \
          torch.sum((flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2) / 2
    ret = ret / np.prod(flow.shape[1:])

    return ret


def regularize_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    # dxy = torch.mul(dxy, dxy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0




 