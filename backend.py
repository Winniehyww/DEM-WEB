import argparse, time, json, os, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_def import UNet, TinyUNet, make_tri_faces, init_identity_model, optimize_refinement_weight, optimize_refinement_weight_ft, compute_mapping_quality, compute_density_loss_triangular, smooth_blend

torch.backends.cudnn.enabled = False  # Disable cuDNN 

def main(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['original','mapped'], required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--ft_ckpt', type=str, required=True)
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

    fn_str = args.fn.replace('^', '**')  
    safe_dict = {
        'x': X, 'y': Y, 'torch': torch, 'np': np,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'pi': np.pi,
        'sqrt': np.sqrt, 'log': np.log,
        'abs': np.abs, 'max': np.max, 'min': np.min,
        'smooth_blend': smooth_blend,
        'pow': lambda a,b: a**b
    }
    p = eval(fn_str, safe_dict)
    
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

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    ft_model = TinyUNet().to(device)
    X = X.to(device)
    Y = Y.to(device)
    p = p.to(device)

    init_identity_model(model)
    ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=5e-3, weight_decay=1e-5)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))


    if args.mode=='mapped':
        ft_model.load_state_dict(torch.load(args.ft_ckpt, map_location=device, weights_only=True))
        # Fine-tune model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        ft_model.train()
        for param in ft_model.parameters():
            param.requires_grad = True

        start = time.time()
        if args.ft_epochs>0:
            for epoch in range(args.ft_epochs):
                # Forward pass
                with torch.no_grad():
                    pos_input = torch.stack([X, Y, p], dim=0).unsqueeze(0)  # [1, 3, N, N]
                    phi_init = model(pos_input)[0]  # [1, 2, N, N]
                    u_test = X + X * (1 - X) * phi_init[0]
                    v_test = Y + Y * (1 - Y) * phi_init[1]

                ft_in = torch.stack([u_test, v_test, p], dim=0).unsqueeze(0)
                ft_phi = ft_model(ft_in)[0]
                u_ft = u_test + u_test * (1 - u_test) * ft_phi[0]
                v_ft = v_test + v_test * (1 - v_test) * ft_phi[1]

                # Triangular density loss on training grid
                density_loss_ft = compute_density_loss_triangular(
                    X, Y, p, u_ft, v_ft, faces, dx
                )

                # Beltrami
                u_ft_x = (u_ft[2:,1:-1]   - u_ft[:-2,1:-1]) / (2*dx)
                u_ft_y = (u_ft[1:-1,2:]   - u_ft[1:-1,:-2]) / (2*dx)
                v_ft_x = (v_ft[2:,1:-1]   - v_ft[:-2,1:-1]) / (2*dx)
                v_ft_y = (v_ft[1:-1,2:]   - v_ft[1:-1,:-2]) / (2*dx)
                fx_ft = u_ft_x + 1j * v_ft_x
                fy_ft = u_ft_y + 1j * v_ft_y
                fz_ft  = 0.5*(fx_ft - 1j*fy_ft)
                fzb_ft = 0.5*(fx_ft + 1j*fy_ft)
                mu_ft = fzb_ft / (fz_ft + 1e-8)
                beltrami_ft = (mu_ft.abs()**2).max()

                lambda_bc = 0.03

                # Combine training density loss with Beltrami
                loss_ft = density_loss_ft + lambda_bc * beltrami_ft

                # Backward pass and optimization
                ft_model.zero_grad()
                loss_ft.backward()
                torch.nn.utils.clip_grad_norm_(ft_model.parameters(), max_norm=1.0)
                ft_optimizer.step()

            ft_model.eval()

            # golden-section optimize
            alpha, _ = optimize_refinement_weight(model, X.to(device), Y.to(device), p.to(device),
                                                torch.from_numpy(faces).to(device), dx)
            alpha_ft, _ = optimize_refinement_weight_ft(model, ft_model, X.to(device), Y.to(device), p.to(device),
                                                torch.from_numpy(faces).to(device), dx)

            # final mapping
            with torch.no_grad():
                phi = model(torch.stack([X,Y,p],dim=0).unsqueeze(0).to(device))[0]
                u = X + X*(1-X)*phi[0]*alpha
                v = Y + Y*(1-Y)*phi[1]*alpha
                ft_in = torch.stack([u, v, p], dim=0).unsqueeze(0)
                ft_phi = ft_model(ft_in)[0]
                u = u + u*(1-u)*ft_phi[0]*alpha_ft
                v = v + v*(1-v)*ft_phi[1]*alpha_ft
        else:
            alpha, _ = optimize_refinement_weight(model, X.to(device), Y.to(device), p.to(device),
                                                torch.from_numpy(faces).to(device), dx)
            with torch.no_grad():
                phi = model(torch.stack([X,Y,p],dim=0).unsqueeze(0).to(device))[0]
                u = X + X*(1-X)*phi[0]*alpha
                v = Y + Y*(1-Y)*phi[1]*alpha

        elapsed = time.time()-start

        # plot mapped domain
        plt.figure(figsize=(16,12))
        plt.tripcolor(u.cpu().numpy().ravel(), v.cpu().numpy().ravel(), faces, p.cpu().numpy().ravel(), shading='flat', cmap='jet')
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
        # Normalize
        orig_rho_norm = orig_rho / orig_rho.mean()
        map_rho_norm  = map_rho  / map_rho.mean()
        # Make bins symmetric around 1
        all_rho = np.concatenate([orig_rho_norm, map_rho_norm])
        max_dist = max(1 - all_rho.min(), all_rho.max() - 1)
        x_min = 1 - max_dist
        x_max = 1 + max_dist
        bins = np.linspace(x_min, x_max, 51)  # 50 bins, symmetric around 1

        # Compute histograms for y-limit
        orig_hist, _ = np.histogram(orig_rho_norm, bins=bins)
        map_hist, _  = np.histogram(map_rho_norm,  bins=bins)
        y_max = max(orig_hist.max(), map_hist.max()) * 1.05  # add margin

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(24,8))
        ax1.hist(orig_rho_norm, bins=bins, edgecolor='k')
        ax1.set(title='Original ρ/mean', xlim=(x_min, x_max), ylim=(0, y_max))
        ax1.axvline(1, color='r', linestyle='--', lw=2)
        ax2.hist(map_rho_norm, bins=bins, edgecolor='k')
        ax2.set(title='Mapped ρ/mean', xlim=(x_min, x_max), ylim=(0, y_max))
        ax2.axvline(1, color='r', linestyle='--', lw=2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out,'hist_compare.png'), dpi=300)

if __name__ == "__main__":
    main()