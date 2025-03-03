import torch
from torch.optim.optimizer import Optimizer

class DPLinearMomentumOptimizer(Optimizer):
    '''
    This optimizer maintains a history of past parameter values and gradients, then applies
    both per-sample momentum (inner momentum) and update-based linear momentum (outer momentum)
    to improve optimization performance while maintaining privacy guarantees.
    
    Args:
        params: Model parameters to optimize
        optimizer: Base optimizer to use (e.g., SGD)
        inner_k0: Number of historical gradients to use for inner momentum
        inner_gamma: Decay factor for inner momentum weights
        a: Low-pass filter coefficients
        b: Low-pass filter coefficients
    '''
    def __init__(self, params, optimizer: Optimizer, inner_k0=5, inner_gamma=0.1,
                 a=None, b=None):
        self.optimizer = optimizer
        self.k0 = inner_k0
        # per-sample momentum weights calculation
        self.inner_gamma = inner_gamma
        weights = [self.inner_gamma ** (inner_k0-1-i) for i in range(inner_k0)]
        weights_sum = sum(weights)
        self.weights = [w/weights_sum for w in weights]
        self.a = a
        self.b = b
        
        #sum(b) - sum(a[1:]) = 1
        filter_sum = sum(b) - sum(a[1:])
        assert abs(filter_sum - 1.0) < 1e-6, f"Linear filter constraint must be 1.0, got {filter_sum:.6f}"
        
        print(f"Inner weights: {[f'{w:.4f}' for w in self.weights]}")
        print(f"Linear filter a coefficients: {[f'{x:.4f}' for x in a]}")
        print(f"Linear filter b coefficients: {[f'{x:.4f}' for x in b]}")
        print(f"Linear filter constraint: {filter_sum:.6f}")
        
        defaults = dict()
        super(DPLinearMomentumOptimizer, self).__init__(params, defaults)
        
        # initialization
        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'param_history'):
                    p.param_history = []
                    for _ in range(self.k0 - 1): 
                        p.param_history.append(p.data.clone())

    def prestep(self, closure):
        """
        Computes gradients for the current parameters and historical parameter values,
        then combines them using weighted averaging based on inner momentum settings.
        
        This function is called before the main optimization step to prepare gradients
        with momentum incorporated at the per-sample level.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            loss: The current loss value
        """
        loss = None
        inner_grads = []

        with torch.enable_grad():
            current_loss = closure()
            
        current_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    current_grads.append(p.grad.clone())
                    p.grad = None
        
        current_params = {}
        for group in self.param_groups:
            for p in group['params']:
                current_params[p] = p.data.clone()
        
        for i in range(self.k0 - 1):
            for group in self.param_groups:
                for p in group['params']:
                    p.data.copy_(p.param_history[i])
            
            with torch.enable_grad():
                loss = closure()
            
            cur_grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        cur_grads.append(p.grad.clone())
                        p.grad = None
            inner_grads.append(cur_grads)
        
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(current_params[p])
        
        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                weighted_grad = torch.zeros_like(p.data)
                weighted_grad += self.weights[0] * current_grads[p_idx]
                for i in range(self.k0 - 1):
                    weighted_grad += self.weights[i+1] * inner_grads[i][p_idx]
                p.grad = weighted_grad
        
        return loss

    def step(self, closure=None):
        """
        Performs a single optimization step using the gradients computed in prestep.
        
        This method applies the low-pass filter defined by 
        coefficients a and b, and then calls the base optimizer's step method.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            loss: The loss value returned by the base optimizer
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                d_p = p.grad

                if len(state) == 0:
                    state['bt'] = torch.zeros(len(self.b)).to(d_p)
                    state['bt'][0] = 1
                    
                state['bt'] = torch.cat((torch.tensor([1]).to(d_p), state['bt'][:-1]))
                norm_factor = torch.inner(torch.tensor(self.b).to(state['bt']), state['bt'])

                if len(self.b) > 1:
                    if 'g_tau' not in state:
                        size = [len(self.b)-1, d_p.numel()]
                        state['g_tau'] = torch.zeros(size, dtype=d_p.dtype).to(d_p)
                        state['g_tau'][0] = d_p.reshape(-1).clone()
                        d_p.mul_(self.b[0])
                    else:
                        G_temp = d_p.reshape(1,-1).clone()
                        d_p.mul_(self.b[0])
                        d_p.add_(torch.einsum('i,ij->j', 
                                torch.tensor(self.b[1:]).to(d_p), 
                                state['g_tau']).reshape(d_p.size()))
                        state['g_tau'] = torch.cat((G_temp, state['g_tau'][:-1]))
                else:
                    d_p.mul_(self.b[0])
                if len(self.a) > 1:
                    if 'm_tau' not in state:
                        size = [len(self.a)-1, d_p.numel()]
                        state['m_tau'] = torch.zeros(size, dtype=d_p.dtype).to(d_p)
                        state['at'] = torch.zeros(len(self.a)-1).to(d_p)
                    else:
                        d_p.add_(torch.einsum('i,ij->j', 
                                torch.tensor(self.a[1:]).to(d_p), 
                                state['m_tau']).reshape(d_p.size()), 
                                alpha=-1)
                        norm_factor -= torch.inner(torch.tensor(self.a[1:]).to(state['at']), 
                                                state['at'])
                    state['at'] = torch.cat((norm_factor.reshape(-1), state['at'][:-1]))
                    state['m_tau'] = torch.cat((d_p.reshape(1,-1).clone(), state['m_tau'][:-1]))

                p.grad.copy_(d_p.div(norm_factor))

            for p in group['params']:
                if len(p.param_history) >= self.k0:
                    p.param_history.pop(0)
                p.param_history.append(p.data.clone())
            
        return self.optimizer.step()