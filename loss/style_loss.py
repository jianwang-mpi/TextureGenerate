import torch
import torch.nn as nn
def gram_matrix(img):
    a, b, c, d = img.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = img.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, img, target_img):
        G = gram_matrix(img)
        T = gram_matrix(target_img)
        loss = self.loss(G, T)
        return loss