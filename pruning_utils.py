# This file will contain helper functions related to the pruning process, including any specialized pruning functions and the SparseGPT functionality.
# DISCLAIMER: The SparseGPT class is a modified version of the original SparseGPT class. The original SparseGPT class can be found in [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot].

import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

# turned this flag to be True
DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class SparseGPT_OPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.batch_inp = []
        self.batch_out = []

    def add_batch(self, inp, out, name, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        ###### added code
        if name == 'fc1' or name == 'fc2':
            self.batch_inp.append(inp[0].clone().detach())
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        ######
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        # del self.H 
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = W[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
            # print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    def fasterprune_vacuum(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01,
        n_vac=3, lmbda=0, cooking_iters=0, lr_vac=0
    ):
        """
        THE CHAMPION VERSION: Aggressive Contrast + Redundancy Mapping.
        This version integrates the n_vac=3 success with Feature Uniqueness.
        """
        # 1. SETUP (Float32 is mandatory for 127-range results)
        W = self.layer.weight.data.clone().float()
        H = self.H.float()
        tick = time.time()

        # 2. FEATURE UNIQUENESS (The tie-breaker)
        # We look at the correlation matrix of inputs to find unique features
        d = torch.diag(H)
        # Correlation C_ij = H_ij / sqrt(H_ii * H_jj)
        C = H / (torch.sqrt(torch.outer(d, d)) + 1e-9)
        # uniqueness = 1 / log(total correlation)
        # This rewards weights that are the 'only ones' sending a specific signal
        uniqueness = 1.0 / torch.log1p(torch.sum(torch.abs(C), dim=1) + 1.0)

        del C
        uniqueness = (uniqueness / uniqueness.max()).reshape((1, -1))
        # Keep the uniqueness nudge subtle (0.9 to 1.0)
        uniqueness = torch.clamp(uniqueness, min=0.9)

        # 3. AGGRESSIVE ROW-WISE VACUUM (Your n_vac=3 discovery)
        row_max = torch.max(torch.abs(W), dim=1, keepdim=True)[0] + 1e-9
        # n_vac=3 creates the w^7 contrast which gave you the 132 PPL
        v_multiplier = torch.pow(torch.abs(W) / row_max, n_vac)

        # 4. PREPARE HESSIAN
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        h_inv_diag = torch.diag(Hinv).reshape((1, -1))
            
        # 5. THE INTEGRATED WINNING SCORE
        # Formula: Standard OBS * Vacuum Contrast * Uniqueness Bonus
        base_score = W**2 / (h_inv_diag + 1e-9)
        importance_scores = base_score * v_multiplier * uniqueness
        del base_score , v_multiplier , uniqueness
        # --- NEW OPTIMIZATION: Convert scores to a small Boolean mask ---
        thresh = torch.sort(importance_scores.flatten())[0][int(importance_scores.numel() * sparsity)]
        global_mask = importance_scores > thresh
        del importance_scores # <--- GIANT SAVINGS: Deletes scores before the loop starts
        
        # 6. EXACT SparseGPT EXECUTION LOOP
        W[:, torch.diag(H) == 0] = 0
        del H    
        Hinv_cholesky = torch.linalg.cholesky(Hinv, upper=True)
        del Hinv
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv_cholesky[i1:i2, i1:i2]
            
            # Mask selection using the integrated champion scores
            mask1 = ~global_mask[:, i1:i2] 

            for i in range(count):
                w = W1[:, i]; d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0 
                Q1[:, i] = q

                # Numerical correction to fix the error of killed weights
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv_cholesky[i1:i2, i2:])
            del W1, Hinv1, mask1
        del Hinv_cholesky, global_mask
        # 7. CONVERT BACK
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        print(f"Champion Vacuum Pruning (n={n_vac}) Done.")
    
        
    def hcv_joint_fastpruner(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01,
        n_vac=2
    ):
        """
        NEW INVENTION: Active-Signal Manifold Pruning (ASMP).
        Combines Teacher's SVD logic with Data-Aware Hessian weighting.
        Target: Beating the 117 PPL baseline.
        """
        # 1. SETUP
        W = self.layer.weight.data.clone().float()
        H = self.H.float()
        dev = self.dev
        tick = time.time()

        # 2. ACTIVE SIGNAL DISCOVERY (The Teacher + Data Hybrid)
        # We look at the weight's importance weighted by the Input Power (diag of H)
        d_diag = torch.diag(H).abs()
        # Scale weights by the signal they actually carry
        # W_active represents the 'Real Information Flow'
        W_active = W * torch.sqrt(d_diag + 1e-9).reshape(1, -1)
        
        # SVD on the Active Signal to find the Principal Reasoning Manifold
        with torch.no_grad():
            # We use the top 25% of singular values to define the 'Signal'
            U, S, Vh = torch.linalg.svd(W_active, full_matrices=False)
            s_mask = torch.zeros_like(S)
            k = max(1, len(S) // 4) 
            s_mask[:k] = 1.0 # Keep only the top logical channels
            
            # Reconstruct the 'Logic Skeleton'
            W_manifold = (U * (S * s_mask).unsqueeze(0)) @ Vh
            # The 'Manifold Survival Score'
            # Weights that align with the Active Manifold get a huge bonus
            manifold_nudge = W_manifold.abs()
            del U, S, Vh, W_manifold, s_mask, W_active

        # 3. HESSIAN PREPARATION
        damp = percdamp * torch.mean(d_diag)
        diag = torch.arange(self.columns, device=dev)
        H_tmp = H.clone()
        H_tmp[diag, diag] += damp
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H_tmp))
        h_inv_diag = torch.diag(Hinv).reshape((1, -1))
        del H_tmp

        # 4. THE ASMP CHAMPION SCORE
        # base = Standard SparseGPT (The 117 PPL foundation)
        base_score = W**2 / (h_inv_diag + 1e-9)
        
        # We use the Vacuum to 'clean' the manifold nudge
        # This creates a sharp gap between 'Logic' and 'Noise'
        max_m = manifold_nudge.max() + 1e-12
        v_nudge = torch.pow(manifold_nudge / max_m, n_vac)
        
        # FINAL FORMULA: Base * (Active Manifold)^0.2
        # A 0.2 power ensures the manifold logic guides the decision 
        # but the Hessian safety math stays in control.
        importance = base_score * torch.pow(v_nudge + 1e-12, 0.2)
        
        # Create Global Mask
        thresh = torch.sort(importance.flatten())[0][int(importance.numel() * sparsity)]
        global_mask = importance > thresh
        del importance, v_nudge, manifold_nudge

        # 5. EXACT SURGERY (The correction)
        W_orig = W.clone()
        Hinv_cholesky = torch.linalg.cholesky(Hinv, upper=True)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone(); Hinv1 = Hinv[i1:i2, i1:i2]
            mask1 = ~global_mask[:, i1:i2] 
            for i in range(count):
                w = W1[:, i]; d = Hinv1[i, i]
                q = w.clone(); q[mask1[:, i]] = 0 
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                W[:, i1+i] = q
            W[:, i2:] -= (W[:, i1:i2] - W1) @ Hinv_cholesky[i1:i2, i2:]
            torch.cuda.empty_cache()

        # 6. CONVERT BACK
        if isinstance(self.layer, transformers.Conv1D): W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        print(f"  ASMP Pruning Done. Active Manifold Protected.")
        
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

def outer_product(a, b):
    return torch.matmul(a.unsqueeze(1), b.unsqueeze(0))
        
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()




class SparseGPT_LlaMA:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.batch_inp = []
        self.batch_out = []

    def add_batch(self, inp, out, name, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        ###### added code
        if name == 'mlp.up_proj' or name == 'mlp.down_proj':
            self.batch_inp.append(inp[0].clone().detach())
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        if name == 'mlp.gate_proj':   # for this layer, we only store the outputs. for inputs, they are shared with 'mlp.up_proj'
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        ######
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        # del self.H 
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = W[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
            # print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
