from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
from torch import distributions
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import sys

class Solver:

    def __init__(self, loader):
        
        self.loader = loader

        self.c_dim = 4
        
        self.lambda_cls = 10.0
        self.lambda_rec = 10.0
        self.lambda_gp = 10.0

        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.n_critic = 6
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.smooth_beta = 0.999
        
        self.model_save_step = 1000
        self.lr_update_step = 1000

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_size = 128

        self.num_iters = 200000
        self.num_iters_decay = 100000

        self.log_step = 10
        self.sample_step = 100

        # Directories.
        self.log_dir = "log"
        self.sample_dir = "sample"
        self.model_save_dir = "model"
        self.result_dir = "resule"

        self.build_model()

    def build_model(self):
        self.G = Generator(conv_dim=64, c_dim=self.c_dim)
        self.G_test = Generator(conv_dim=64, c_dim=self.c_dim)
        self.D = Discriminator(self.image_size, 64, self.c_dim)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        #self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.g_lr, alpha=0.99, eps=1e-8)
        #self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.d_lr, alpha=0.99, eps=1e-8)
        
        self.G.to(self.device)
        self.G_test.to(self.device)
        self.D.to(self.device)

        self.update_average(self.G_test, self.G, 0.)

    def eval_model(self):
        self.G.eval()
        self.G_test.eval()
        self.D.eval()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        G_test_path = os.path.join(self.model_save_dir, '{}-G_test.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G_test.load_state_dict(torch.load(G_test_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def update_average(self, model_tgt, model_src, beta):
        toogle_grad(model_src, False)
        toogle_grad(model_tgt, False)

        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    def get_zdist(self, dist_name, dim, device=None):
        # Get distribution
        if dist_name == 'uniform':
            low = -torch.ones(dim, device=device)
            high = torch.ones(dim, device=device)
            zdist = distributions.Uniform(low, high)
        elif dist_name == 'gauss':
            mu = torch.zeros(dim, device=device)
            scale = torch.ones(dim, device=device)
            zdist = distributions.Normal(mu, scale)
        else:
            raise NotImplementedError

        # Add dim attribute
        zdist.dim = dim

        return zdist

    def getBatch(self):
        try:
            x_real, label_org = next(self.data_iter)
        except:
            while True:
                try:
                    self.data_iter = iter(self.loader)
                    x_real, label_org = next(self.data_iter)
                    break
                except:
                    pass
        return x_real, label_org

    def train(self, start_iter=0):

        g_lr = self.g_lr
        d_lr = self.d_lr

        zdist = None

        BCELoss = torch.nn.BCELoss()

        if start_iter > 0:
            self.restore_model(start_iter)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iter, self.num_iters):
            
            x_real, label_org = self.getBatch()

            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            # input images
            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            if zdist is None:
                zdist = self.get_zdist("uniform", (3,x_real.size(2),x_real.size(3)), device=self.device)

            # make noise
            noise = zdist.sample((x_real.size(0),))

            

            if (i) % 1 == 0:
                # train discriminator
                toogle_grad(self.G, False)
                toogle_grad(self.D, True)
                
                out_src, out_cls = self.D(x_real)
                label_real = torch.full((x_real.size(0),1), 1.0, device=self.device)
                #d_loss_real = BCELoss(out_src.view(x_real.size(0),1), label_real)
                #print(out_src,d_loss_real)
                d_loss_real = -torch.mean(out_src)
                d_loss_cls_real = self.classification_loss(out_cls, label_org)

                x_mask = self.G(x_real, c_trg)
                x_fake = x_mask * x_real + (1.0-x_mask) * noise
                out_src_fake, out_cls_fake = self.D(x_fake.detach())

                label_fake = torch.full((x_real.size(0),1), 0.0, device=self.device)
                #d_loss_fake = BCELoss(out_src.view(x_real.size(0),1), label_fake)

                
                d_loss_cls_fake = self.classification_loss(out_cls_fake, label_org)
                d_loss_fake = torch.mean(out_src_fake)

                # gp_loss
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1.0 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                d_loss =d_loss_real + d_loss_fake + self.lambda_cls * (d_loss_cls_real+d_loss_cls_fake) + self.lambda_gp * d_loss_gp
                
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls_real'] = d_loss_cls_real.item()
                loss['D/loss_cls_fake'] = d_loss_cls_fake.item()
                loss['D/loss_gp'] = d_loss_gp.item()

            # train generator
            if (i+1) % self.n_critic == 0:
                toogle_grad(self.G, True)
                toogle_grad(self.D, False)
                x_mask = self.G(x_real, c_trg)
                x_fake = x_mask * x_real + (1.0-x_mask) * noise
                out_src, out_cls = self.D(x_fake)

                label_real = torch.full((x_real.size(0),1), 1.0, device=self.device)
                #g_loss_fake = BCELoss(out_src.view(x_real.size(0),1), label_real)
                
                
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(-out_cls+1.0, label_org)

                # Target-to-original domain.
                #x_reconst = self.G(x_fake, c_org)
                #g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # backward
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls
                #g_loss = self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # smoothing
                self.update_average(self.G_test, self.G, self.smooth_beta)


                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                #loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                #if self.use_tensorboard:
                #    for tag, value in loss.items():
                #        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                x_fake_list = [x_real]
                x_fake_list.append(x_fake)
                #x_fake_list.append(x_reconst)
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    

                """
                with torch.no_grad():

                    data_iter_test = iter(self.loader)
                    x_real, label_org = next(data_iter_test)

                    rand_idx = torch.randperm(label_org.size(0))
                    label_trg = label_org[rand_idx]

                    label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
                    
                    y_fake = self.G(x_real, label_trg)
                    
                
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
                """

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                G_test_path = os.path.join(self.model_save_dir, '{}-G_test.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.G_test.state_dict(), G_test_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay lr
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self, test_iters=None):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        if test_iters is not None:
            self.restore_model(test_iters)

        #self.eval_model()
            
        # Set data loader.
        data_loader = self.loader
            
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = []
                for j in range(self.c_dim):
                    c_trg = c_org.clone()
                    c_trg[:,:] = 0.0
                    c_trg[:,j] = 1.0
                    c_trg_list.append(c_trg.to(self.device))
                
                # Translate images.
                x_fake_list = []
                
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G_test(x_real, c_trg))
                print(x_fake_list[0])

                # Save the translated images.
                try:
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
                except:
                    import traceback
                    traceback.print_exc()
                    print('Error {}...'.format(result_path))
                

# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
