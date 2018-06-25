_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.alpha = 0.001
_default_hyperparam.beta1 = 0.9
_default_hyperparam.beta2 = 0.999
_default_hyperparam.beta3 = 0.999
_default_hyperparam.eps = 1e-8
_default_hyperparam.eta = 1.0
_default_hyperparam.lower_threshold = 0.1
_default_hyperparam.upper_threshold=10
_default_hyperparam.weight_decay_rate = 0
_default_hyperparam.amsgrad = False


def _learning_rate(hp, t):
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Adam optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    return hp.alpha * math.sqrt(fix2) / fix1


class EveRule(optimizer.UpdateRule):

    def __init__(self, parent_hyperparam=None,
                 alpha=None, beta1=None, beta2=None, beta3=None, eps=None,
                 eta=None, lower_threshold=None, upper_threshold=None, weight_decay_rate=None, amsgrad=None):
        super(EveRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if beta1 is not None:
            self.hyperparam.beta1 = beta1
        if beta2 is not None:
            self.hyperparam.beta2 = beta2
        if beta3 is not None:
            self.hyperparam.beta3 = beta3
        if eps is not None:
            self.hyperparam.eps = eps
        if eta is not None:
            self.hyperparam.eta = eta
        if lower_threshold is not None:
            self.hyperparam.lower_threshold = lower_threshold
        if upper_threshold is not None:
            self.hyperparam.upper_threshold = upper_threshold
        if weight_decay_rate is not None:
            self.hyperparam.weight_decay_rate = weight_decay_rate
        if amsgrad is not None:
            self.hyperparam.amsgrad = amsgrad

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)
            self.state['d'] = xp.ones(1, dtype=param.data.dtype)
            self.state['f'] = xp.zeros(1, dtype=param.data.dtype)
            if self.hyperparam.amsgrad:
                self.state['vhat'] = xp.zeros_like(param.data)
                
    def _update_d_and_f(self):
        d, f = self.state['d'], self.state['f']
        
        global global_loss
        
        #print(g_loss)
        
        if self.t > 1:
            old_f = float(cuda.to_cpu(self.state['f']))
            
            if old_f >= 0:
                if global_loss > old_f:
                    delta = self.hyperparam.lower_threshold + 1.
                    Delta = self.hyperparam.upper_threshold + 1.
                else:
                    delta = 1. / (self.hyperparam.upper_threshold + 1.)
                    Delta = 1. / (self.hyperparam.lower_threshold + 1.)
                    
            else:
                
                if global_loss > old_f:
                    delta = 1. / (self.hyperparam.lower_threshold + 1.)
                    Delta = 1. / (self.hyperparam.upper_threshold + 1.)
                    
                else:
                    delta = self.hyperparam.upper_threshold + 1.
                    Delta = self.hyperparam.lower_threshold + 1.
            delta*=old_f
            Delta*=old_f
            new_f = min(max(delta, global_loss), Delta)
            
            if old_f >= 0:
                if global_loss > old_f:
                    r = (new_f - old_f) / (old_f + 1e-12)
                else:
                    r = (old_f - new_f) / (new_f + 1e-12)
            else:
                if global_loss > old_f:
                    r = (old_f - new_f) / (new_f - 1e-12)
                else:
                    r = (new_f - old_f) / (old_f - 1e-12)
            
            d += (1 - self.hyperparam.beta3) * (r - d)
            #c = min(max(delta, global_loss / (old_f + 1e-12)), Delta)
            #new_f = c * old_f
            #r = abs(new_f - old_f) / (min(new_f, old_f) + 1e-12)
            #d += (1 - self.hyperparam.beta3) * (r - d)
            f[:] = new_f
        else:
            f[:] = global_loss

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        m, v, d = self.state['m'], self.state['v'], self.state['d']
        
        self._update_d_and_f()

        m += (1 - hp.beta1) * (grad - m)
        v += (1 - hp.beta2) * (grad * grad - v)

        if hp.amsgrad:
            vhat = self.state['vhat']
            numpy.maximum(vhat, v, out=vhat)
        else:
            vhat = v
        param.data -= hp.eta * (self.lr * m / (d * numpy.sqrt(vhat) + hp.eps) +
                                hp.weight_decay_rate * param.data)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
            
        self._update_d_and_f()
        
        if hp.amsgrad:
            cuda.elementwise(
                'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps, \
                 T eta, T weight_decay_rate',
                'T param, T m, T v, T d, T vhat',
                '''m += one_minus_beta1 * (grad - m);
                   v += one_minus_beta2 * (grad * grad - v);
                   vhat = max(vhat, v);
                   param -= eta * (lr * m / (d * sqrt(vhat) + eps) +
                                   weight_decay_rate * param);''',
                'adam')(grad, self.lr, 1 - hp.beta1,
                        1 - hp.beta2, hp.eps,
                        hp.eta, hp.weight_decay_rate,
                        param.data, self.state['m'], self.state['v'], self.state['d'],
                        self.state['vhat'])
        else:
            cuda.elementwise(
                'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps, \
                 T eta, T weight_decay_rate',
                'T param, T m, T v, T d',
                '''m += one_minus_beta1 * (grad - m);
                   v += one_minus_beta2 * (grad * grad - v);
                   param -= eta * (lr * m / (d * sqrt(v) + eps) +
                                   weight_decay_rate * param);''',
                'adam')(grad, self.lr, 1 - hp.beta1,
                        1 - hp.beta2, hp.eps,
                        hp.eta, hp.weight_decay_rate,
                        param.data, self.state['m'], self.state['v'],  self.state['d'])

    @property
    def lr(self):
        return _learning_rate(self.hyperparam, self.t)
    
    """
    def update(self, loss=None):
        # Overwrites GradientMethod.update in order to get loss values
        if loss is None:
            raise RuntimeError('Eve.update requires lossfun to be specified')
            
        print(1, loss.shape)
        print(2, *args)
        print(3, **kwds)
        #loss_var = lossfun(*args, **kwds)
        self.loss = float(loss)
        super(Eve, self).update(lossfun=lambda: loss_var)
    """

class Eve(optimizer.GradientMethod):


    def __init__(self,
                 alpha=_default_hyperparam.alpha,
                 beta1=_default_hyperparam.beta1,
                 beta2=_default_hyperparam.beta2,
                 beta3=_default_hyperparam.beta3,
                 eps=_default_hyperparam.eps,
                 eta=_default_hyperparam.eta,
                 lower_threshold=_default_hyperparam.lower_threshold,
                 upper_threshold=_default_hyperparam.upper_threshold,
                 weight_decay_rate=_default_hyperparam.weight_decay_rate,
                 amsgrad=_default_hyperparam.amsgrad):
        super(Eve, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.beta3 = beta3
        self.hyperparam.eps = eps
        self.hyperparam.eta = eta
        self.hyperparam.lower_threshold = lower_threshold
        self.hyperparam.upper_threshold = upper_threshold
        self.hyperparam.weight_decay_rate = weight_decay_rate
        self.hyperparam.amsgrad = amsgrad

    alpha = optimizer.HyperparameterProxy('alpha')
    beta1 = optimizer.HyperparameterProxy('beta1')
    beta2 = optimizer.HyperparameterProxy('beta2')
    beta3 = optimizer.HyperparameterProxy('beta3')
    eps = optimizer.HyperparameterProxy('eps')
    eta = optimizer.HyperparameterProxy('eta')
    lower_threshold = optimizer.HyperparameterProxy('lower_threshold')
    upper_threshold = optimizer.HyperparameterProxy('upper_threshold')
    weight_decay_rate = optimizer.HyperparameterProxy('weight_decay_rate')
    amsgrad = optimizer.HyperparameterProxy('amsgrad')

    def create_update_rule(self):
        
        return EveRule(self.hyperparam)

    @property
    def lr(self):
        return _learning_rate(self.hyperparam, self.t)
    
    """
    def update(self, lossfun=None, *args, **kwds):
        Updates parameters based on a loss function or computed gradients.
        This method runs in two ways.
        - If ``lossfun`` is given, then it is used as a loss function to
          compute gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.
        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the update rule of each
        parameter.
        
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward(loss_scale=self._loss_scale)
            del loss

        self.reallocate_cleared_grads()

        self.call_hooks('pre')

        self.t += 1
        for param in self.target.params():
            param.update()

        self.reallocate_cleared_grads()

        self.call_hooks('post')
        
    def setF(self, f):
        self.loss = f
    """
