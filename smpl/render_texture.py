# Create renderer
import chumpy as ch
import numpy as np
from opendr.renderer import TexturedRenderer, ColoredRenderer
# Assign attributes to renderer
from get_body_mesh import get_body_mesh
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
import cv2
from scipy.sparse import csc_matrix
import scipy.sparse as sp


class Renderer:
    def __init__(self, obj_path, model_path, w=224, h=224):
        self.m = get_body_mesh(obj_path, trans=ch.array([0, 0, 4]), rotation=ch.array([np.pi / 2, 0, 0]))
        # Load SMPL model (here we load the female model)
        self.body = load_model(model_path)
        self.w = w
        self.h = h
        self.img_size = min(self.w, self.h)

        self.num_cam = 3
        self.num_theta = 72
        self.num_beta = 10

    def set_texture(self, img_bgr):
        """
        set the texture image for the human body
        :param img_bgr: image should be bgr format
        :return:
        """
        # sz = np.sqrt(np.prod(img_bgr.shape[:2]))
        # sz = int(np.round(2 ** np.ceil(np.log(sz) / np.log(2))))
        self.m.texture_image = img_bgr.astype(np.float64) / 255.
        return self.m

    def render(self, thetas, texture_bgr, rotate=np.array([0, 0, 0]), background_img=None):
        """
        get the rendered image and rendered silhouette
        :param thetas: model parameters, 3 * camera parameter + 72 * body pose + 10 * body shape
        :param texture_bgr: texture image in bgr format
        :return: the rendered image and deviation of rendered image to texture image
        (rendered image, deviation of rendered image, silhouette)
        """
        self.set_texture(texture_bgr)
        thetas = thetas.reshape(-1)
        cams = thetas[:self.num_cam]
        theta = thetas[self.num_cam: (self.num_cam + self.num_theta)]
        beta = thetas[(self.num_cam + self.num_theta):]

        self.body.pose[:] = theta
        self.body.betas[:] = beta

        #
        # size = cams[0] * min(self.w, self.h)
        # position = cams[1:3] * min(self.w, self.h) / 2 + min(self.w, self.h) / 2
        """
        ####################################################################
        ATTENTION!
        I do not know why the flength is 500.
        But it worked
        ####################################################################
        """

        texture_rn = TexturedRenderer()
        texture_rn.camera = ProjectPoints(v=self.body, rt=rotate, t=ch.array([0, 0, 2]),
                                          f=np.ones(2) * self.img_size * 0.62,
                                          c=np.array([self.w / 2, self.h / 2]),
                                          k=ch.zeros(5))
        texture_rn.frustum = {'near': 1., 'far': 10., 'width': self.w, 'height': self.h}
        texture_rn.set(v=self.body, f=self.m.f, vc=self.m.vc, texture_image=self.m.texture_image, ft=self.m.ft,
                       vt=self.m.vt)
        if background_img is not None:
            texture_rn.background_image = background_img / 255. if background_img.max() > 1 else background_img

        silhouette_rn = ColoredRenderer()
        silhouette_rn.camera = ProjectPoints(v=self.body, rt=rotate, t=ch.array([0, 0, 2]),
                                             f=np.ones(2) * self.img_size * 0.62,
                                             c=np.array([self.w / 2, self.h / 2]),
                                             k=ch.zeros(5))
        silhouette_rn.frustum = {'near': 1., 'far': 10., 'width': self.w, 'height': self.h}
        silhouette_rn.set(v=self.body, f=self.m.f, vc=np.ones_like(self.body), bgcolor=np.zeros(3))

        return texture_rn.r, texture_dr_wrt(texture_rn, silhouette_rn.r), silhouette_rn.r


def texture_dr_wrt(texture_rn, clr_im):
    """
    Change original texture dr_wrt
    use the rendered silhouette to avoid holes in the rendered image
    change the output dr from rgb format to bgr format
    :param texture_rn:
    :param clr_im:
    :return:
    """
    IS = np.nonzero(clr_im[:, :, 0].ravel() != 0)[0]
    JS = texture_rn.texcoord_image_quantized.ravel()[IS]

    # if True:
    #     cv2.imshow('clr_im', clr_im)
    #     # cv2.imshow('texmap', texture_rn.texture_image.r)
    #     cv2.waitKey(0)

    r = clr_im[:, :, 0].ravel()[IS]
    g = clr_im[:, :, 1].ravel()[IS]
    b = clr_im[:, :, 2].ravel()[IS]
    data = np.concatenate((b, g, r))

    IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
    JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))

    return sp.csc_matrix((data, (IS, JS)), shape=(texture_rn.r.size, texture_rn.texture_image.r.size))


def bbox(img):
    rows = np.any(img, axis=0)
    cols = np.any(img, axis=1)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


if __name__ == '__main__':
    renderer = Renderer('models/body.obj', 'models/neutral.pkl', w=224, h=224)
    thetas = np.zeros(85)

    thetas[0:3] = 112
    thetas[3] = np.pi
    texture_bgr = cv2.imread('/home/wangjian02/Projects/TextureGAN/tmp/test_img/out_uv_prw/pede.png')
    texture_bgr = cv2.resize(texture_bgr, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

    rn, deviation, silhouette = renderer.render(thetas, texture_bgr, rotate=np.array([0, 0, 0]))
    # # Show it
    rn = (rn * 255.).astype(np.uint8)
    rn = cv2.cvtColor(rn, code=cv2.COLOR_RGB2BGR)


    texture_bgr = cv2.imread('/home/wangjian02/Projects/TextureGAN/tmp/video_avatar/tex-female-1-casual.jpg')
    texture_bgr = cv2.resize(texture_bgr, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

    compare, deviation, silhouette = renderer.render(thetas, texture_bgr, rotate=np.array([0, 0, 0]))
    # # Show it
    compare = (compare * 255.).astype(np.uint8)
    compare = cv2.cvtColor(compare, code=cv2.COLOR_RGB2BGR)
    cv2.imshow('rn1', compare)
    cv2.waitKey(0)
    # cv2.destroyWindow('rn1')
    cv2.imshow('rn2', rn)
    cv2.waitKey(0)

    render_other = False
    if render_other:
        # show silhouette
        silhouette = (silhouette * 255.).astype(np.uint8)
        silhouette = cv2.cvtColor(silhouette, code=cv2.COLOR_RGB2BGR)
        cv2.imshow('silhouette', silhouette)
        cv2.waitKey()

        rmin, rmax, cmin, cmax = bbox(silhouette[:, :, 0])

        texture_bgr = texture_bgr.reshape(-1)

        new_rendered = deviation.dot(texture_bgr.T)
        # new_rendered = new_rendered.toarray()
        new_rendered = np.reshape(new_rendered, [224, 224, 3]).astype(np.uint8)
        new_rendered = cv2.rectangle(new_rendered, (rmin, cmin), (rmax, cmax), color=(0, 0, 255), thickness=2)
        # new_rendered = cv2.inpaint(new_rendered, )
        # new_rendered = cv2.resize(new_rendered, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('new', new_rendered)
        cv2.waitKey()
