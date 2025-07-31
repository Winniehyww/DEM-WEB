import argparse, time, json, os, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_def import UNet, make_tri_faces, init_identity_model, optimize_refinement_weight, compute_mapping_quality, compute_density_loss_triangular, smooth_blend

def main(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['original','mapped'], required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--fn',   type=str, required=True)
    parser.add_argument('--N',    type=int, required=True)
    parser.add_argument('--out',  type=str, required=True)
    parser.add_argument('--ft_epochs', type=int, default=0, required=True)
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Prepare grid and p_test
    N = args.N
    x = torch.linspace(0,1,N)
    y = torch.linspace(0,1,N)
    X, Y = torch.meshgrid(x,y, indexing='ij')
    dx = 1.0/(N-1)
    # safe eval of fn
    p = eval(args.fn, {'x':X,'y':Y,'torch':torch, 
                       'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                       'exp': np.exp, 'pi': np.pi,
                       'sqrt': np.sqrt, 'log': np.log,
                       'abs': np.abs, 'max': np.max, 'min': np.min,
                       'smooth_blend': smooth_blend})
    p = p - p.min() + 0.1
    p = p / (p.sum()*dx*dx)
    faces = make_tri_faces(N)

    # 2. Original domain plot
    plt.figure(figsize=(16,12))
    plt.tripcolor(X.numpy().ravel(), Y.numpy().ravel(), faces, p.numpy().ravel(), shading='flat', cmap='jet')
    plt.triplot(X.numpy().ravel(), Y.numpy().ravel(), faces, color='k', lw=0.1)
    plt.gca().set_aspect('equal')
    plt.colorbar(label='Population')
    plt.savefig(os.path.join(args.out,'original.png'), dpi=300)

    if args.mode=='mapped':
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet().to(device)
        init_identity_model(model)
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
        
        # Fine-tune model
        model.train()
        start = time.time()
        for epoch in range(args.ft_epochs):
            # Forward pass
            pos_input = torch.stack([X, Y, p], dim=0).unsqueeze(0)  # [1, 3, N, N]
            phi_init = model(pos_input)[0]  # [1, 2, N, N]
            u_test = X + X * (1 - X) * phi_init[0]
            v_test = Y + Y * (1 - Y) * phi_init[1]

            # Triangular density loss on training grid
            density_loss_ft = compute_density_loss_triangular(
                X, Y, p, u_test, v_test, faces, dx
            )

            # Beltrami
            u_ft_x = (u_test[2:,1:-1]   - u_test[:-2,1:-1]) / (2*dx)
            u_ft_y = (u_test[1:-1,2:]   - u_test[1:-1,:-2]) / (2*dx)
            v_ft_x = (v_test[2:,1:-1]   - v_test[:-2,1:-1]) / (2*dx)
            v_ft_y = (v_test[1:-1,2:]   - v_test[1:-1,:-2]) / (2*dx)
            fx_ft = u_ft_x + 1j * v_ft_x
            fy_ft = u_ft_y + 1j * v_ft_y
            fz_ft  = 0.5*(fx_ft - 1j*fy_ft)
            fzb_ft = 0.5*(fx_ft + 1j*fy_ft)
            mu_ft = fzb_ft / (fz_ft + 1e-8)
            beltrami_ft = (mu_ft.abs()**2).max()

            lambda_bc = 0.04

            # Combine training density loss with Beltrami
            loss_ft = density_loss_ft + lambda_bc * beltrami_ft

            # Backward pass and optimization
            model.zero_grad()
            loss_ft.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ft_optimizer.step()

        model.eval()

        # golden-section optimize
        alpha, _ = optimize_refinement_weight(model, X.to(device), Y.to(device), p.to(device),
                                             torch.from_numpy(faces).to(device), dx)

        # final mapping
        with torch.no_grad():
            phi = model(torch.stack([X,Y,p],dim=0).unsqueeze(0).to(device))[0]
            u = X + X*(1-X)*phi[0]*alpha
            v = Y + Y*(1-Y)*phi[1]*alpha

        elapsed = time.time()-start

        # plot mapped domain
        plt.figure(figsize=(16,12))
        plt.tripcolor(u.cpu().numpy().ravel(), v.cpu().numpy().ravel(), faces, p.numpy().ravel(), shading='flat', cmap='jet')
        plt.triplot(u.cpu().numpy().ravel(), v.cpu().numpy().ravel(), faces, color='k', lw=0.1)
        plt.gca().set_aspect('equal')
        plt.colorbar(label='Population')
        plt.savefig(os.path.join(args.out,'mapped.png'), dpi=300)

        # compute quality & metrics
        quality = compute_mapping_quality(X, Y, p, u, v, faces, dx)
        metrics = {
            'Time (s)': elapsed,
            'Std/Mean Orig': (quality['density_orig'].std()/quality['density_orig'].mean()).item(),
            'Std/Mean Map':  (quality['density_map'].std()/ quality['density_map'].mean()).item(),
            'Max |μ|':       quality['beltrami_max'],
            'Mean |μ|':      quality['beltrami_mean'],
        }
        with open(os.path.join(args.out,'metrics.json'),'w') as f:
            json.dump(metrics,f,indent=2)

        # histogram comparison
        orig_rho = quality['density_orig'].cpu().numpy().ravel()
        map_rho  = quality['density_map'].cpu().numpy().ravel()
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(24,8))
        ax1.hist(orig_rho/orig_rho.mean(), bins=50, edgecolor='k')
        ax1.set(title='Original ρ/mean', xlim=(-0.5,2.5))
        ax2.hist( map_rho/map_rho.mean(), bins=50, edgecolor='k')
        ax2.set(title='Mapped ρ/mean', xlim=(-0.5,2.5))
        plt.tight_layout()
        plt.savefig(os.path.join(args.out,'hist_compare.png'), dpi=300)

if __name__ == "__main__":
    main()