import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_tri_faces(N: int) -> np.ndarray:
    """
    Create a 2-triangle triangulation for an N x N grid.
    Returns an array of shape (2*(N-1)*(N-1), 3) of vertex indices.
    """
    F_count = 2 * (N - 1) * (N - 1)
    faces = np.zeros((F_count, 3), dtype=np.int64)
    k = 0
    for i in range(N - 1):
        for j in range(N - 1):
            top_left = i * N + j
            faces[k]     = [top_left, top_left + 1,     top_left + N]
            faces[k + 1] = [top_left + 1, top_left + N + 1, top_left + N]
            k += 2
    return faces


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels,  out_channels,  kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels,  base_channels*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*2, base_channels*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*4, base_channels*8))
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels*8, base_channels*16)
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels*8 + base_channels*4, base_channels*8)
        self.up2  = nn.ConvTranspose2d(base_channels*8, base_channels*4,  kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels*4 + base_channels*2, base_channels*4)
        self.up3  = nn.ConvTranspose2d(base_channels*4, base_channels*2,  kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels*2 + base_channels,  base_channels*2)
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # Bottleneck
        x5 = self.bottleneck(x4)
        # Decoding path
        x  = self.up1(x5)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x  = torch.cat([x, x3], dim=1)
        x  = self.dec1(x)
        x  = self.up2(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x  = torch.cat([x, x2], dim=1)
        x  = self.dec2(x)
        x  = self.up3(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x  = torch.cat([x, x1], dim=1)
        x  = self.dec3(x)
        return self.outc(x)


def init_identity_model(model: nn.Module) -> None:
    """
    Initialize last conv layer (out_channels==2) to zero so that model starts as identity.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 2:
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def _pick_gn_groups(c: int) -> int:
    for g in (8, 4, 2):
        if c % g == 0:
            return g
    return 1

class SepConv2d(nn.Module):
    """Depthwise separable 3x3 -> 1x1 conv with reflect padding."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, padding_mode="reflect",
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn = nn.GroupNorm(_pick_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        return self.act(x)

class TinyDoubleSep(nn.Module):
    """Two separable convs, lightweight replacement for DoubleConv."""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            SepConv2d(in_ch, mid_ch),
            SepConv2d(mid_ch, out_ch),
        )
    def forward(self, x) -> torch.Tensor:
        return self.block(x)

class TinyUNet(nn.Module):
    """
    Smaller U-Net (fixed skips + exact output size):
      - Depth = 2 downs / 2 ups
      - Base channels = 8 by default
      - Depthwise-separable convs
      - Bilinear upsample + 1x1 conv
      - Output: Tanh, 2 channels, same HxW as input
    """
    def __init__(self, in_channels=3, out_channels=2, base_channels=4):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc   = TinyDoubleSep(in_channels, c)          # [H,W] -> c
        self.down1 = nn.Sequential(nn.MaxPool2d(2), TinyDoubleSep(c, 2*c))   # -> [H//2,W//2]
        self.down2 = nn.Sequential(nn.MaxPool2d(2), TinyDoubleSep(2*c, 4*c)) # -> [H//4,W//4]

        # Bottleneck (same spatial as down2)
        self.bot = TinyDoubleSep(4*c, 8*c)  # [H//4,W//4]

        # Decoder
        self.up1 = nn.Sequential(  # [H//4 -> H//2]
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(8*c, 4*c, kernel_size=1, bias=False),
        )
        # concat with x2 (2c) -> channels 4c + 2c = 6c
        self.dec1 = TinyDoubleSep(6*c, 4*c)

        self.up2 = nn.Sequential(  # [H//2 -> ~H]
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(4*c, 2*c, kernel_size=1, bias=False),
        )
        # concat with x1 (c) -> channels 2c + c = 3c
        self.dec2 = TinyDoubleSep(3*c, 2*c)

        self.outc = nn.Sequential(
            nn.Conv2d(2*c, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        # Encoder
        x1 = self.inc(x)     # [H,W], c
        x2 = self.down1(x1)  # [H//2,W//2], 2c
        x3 = self.down2(x2)  # [H//4,W//4], 4c

        # Bottleneck
        xb = self.bot(x3)    # [H//4,W//4], 8c

        # Decoder
        y = self.up1(xb)                                   # ~[H//2,W//2], 4c
        if y.shape[2:] != x2.shape[2:]:
            y = F.interpolate(y, size=x2.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, x2], dim=1)                      # 6c
        y = self.dec1(y)                                   # 4c

        y = self.up2(y)                                    # ~[H,W], 2c
        if y.shape[2:] != x1.shape[2:]:
            y = F.interpolate(y, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, x1], dim=1)                      # 3c
        y = self.dec2(y)                                   # 2c

        # Guarantee exact match with input spatial size (handles odd sizes like 65)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

        return self.outc(y)


def relu(x: float) -> float:
    return max(x, 0.0)

def smooth_blend(x: torch.Tensor, center: float, width: float) -> torch.Tensor:
    # logistic transition centered at `center` over `width`
    return 1.0 / (1.0 + torch.exp(-(x - center) / width))

def compute_density_loss_triangular(X, Y, p, u, v, faces, dx) -> torch.Tensor:
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    pf      = p.reshape(-1)
    uf, vf  = u.reshape(-1), v.reshape(-1)
    i1, i2, i3 = faces[:,0], faces[:,1], faces[:,2]

    x1, x2, x3 = Xf[i1], Xf[i2], Xf[i3]
    y1, y2, y3 = Yf[i1], Yf[i2], Yf[i3]
    u1, u2, u3 = uf[i1], uf[i2], uf[i3]
    v1, v2, v3 = vf[i1], vf[i2], vf[i3]

    orig_areas   = 0.5 * torch.abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
    mapped_areas = 0.5 * torch.abs((u2-u1)*(v3-v1) - (v2-v1)*(u3-u1))

    mass    = (pf[i1] + pf[i2] + pf[i3]) / 3 * orig_areas
    density = mass / (mapped_areas + 1e-8)

    # return density.std()/density.mean()

    total_pop = p.sum() * (dx * dx)
    target    = total_pop.expand_as(density)
    return F.smooth_l1_loss(density, target)


def compute_mapping_quality(X: torch.Tensor,
                             Y: torch.Tensor,
                             p: torch.Tensor,
                             u: torch.Tensor,
                             v: torch.Tensor,
                             faces: torch.Tensor,
                             dx: float) -> dict:
    """
    Compute density uniformity, overlap, and Beltrami statistics for the mapping.
    Returns a dict with keys: density_error, overlap_ratio, beltrami_max,
    beltrami_mean, min_jacobian, density_orig, density_map.
    """
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    pf, uf, vf = p.reshape(-1), u.reshape(-1), v.reshape(-1)
    i1, i2, i3 = faces[:,0], faces[:,1], faces[:,2]
    x1, x2, x3 = Xf[i1], Xf[i2], Xf[i3]
    y1, y2, y3 = Yf[i1], Yf[i2], Yf[i3]
    u1, u2, u3 = uf[i1], uf[i2], uf[i3]
    v1, v2, v3 = vf[i1], vf[i2], vf[i3]
    # triangle areas
    orig_areas   = 0.5 * torch.abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
    mapped_areas = 0.5 * torch.abs((u2 - u1)*(v3 - v1) - (v2 - v1)*(u3 - u1))
    # density
    mass         = (pf[i1] + pf[i2] + pf[i3]) / 3 * orig_areas
    density_orig = mass / (orig_areas   + 1e-10)
    density_map  = mass / (mapped_areas + 1e-10)
    target       = p.sum() * (dx * dx)
    density_error = F.smooth_l1_loss(density_map, target.expand_as(density_map))
    # Jacobian and overlap
    u_x = (u[2:,1:-1] - u[:-2,1:-1]) / (2 * dx)
    u_y = (u[1:-1,2:] - u[1:-1,:-2]) / (2 * dx)
    v_x = (v[2:,1:-1] - v[:-2,1:-1]) / (2 * dx)
    v_y = (v[1:-1,2:] - v[1:-1,:-2]) / (2 * dx)
    jac = u_x * v_y - u_y * v_x
    overlap_ratio = (jac < 0).float().mean().item()
    # Beltrami
    fx  = u_x + 1j * v_x
    fy  = u_y + 1j * v_y
    fz  = 0.5 * (fx - 1j * fy)
    fzb = 0.5 * (fx + 1j * fy)
    mu  = fzb / (fz + 1e-8)
    beltrami_max  = mu.abs().max().item()
    beltrami_mean = mu.abs().mean().item()
    min_jacobian  = jac.min().item()
    return {
        'density_error': density_error.item(),
        'overlap_ratio': overlap_ratio,
        'beltrami_max':  beltrami_max,
        'beltrami_mean': beltrami_mean,
        'min_jacobian':  min_jacobian,
        'density_orig':  density_orig,
        'density_map':   density_map
    }


def optimize_refinement_weight(model: nn.Module,
                               X: torch.Tensor,
                               Y: torch.Tensor,
                               p: torch.Tensor,
                               faces: torch.Tensor,
                               dx: float,
                               lb: float = 0.1,
                               ub: float = 3.0,
                               tol: float = 1e-4,
                               max_iter: int = 50) -> tuple:
    """
    Golden‑section search for refinement weight alpha that
    minimizes density_error + penalty for Beltrami > 1.
    Returns (alpha_opt, loss_opt).
    """
    # initial mapping
    uv = model(torch.stack([X, Y, p], dim=0).unsqueeze(0))[0]
    u_init = X + X * (1 - X) * uv[0]
    v_init = Y + Y * (1 - Y) * uv[1]
    m_init = compute_mapping_quality(X, Y, p, u_init, v_init, faces, dx)
    a, b = lb, ub
    if m_init['beltrami_max'] >= 1:
        b = 1.0
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    def loss_at(alpha):
        uv = model(torch.stack([X, Y, p], dim=0).unsqueeze(0))[0]
        u = X + X*(1-X) * uv[0] * alpha
        v = Y + Y*(1-Y) * uv[1] * alpha
        q = compute_mapping_quality(X, Y, p, u, v, faces, dx)
        return q['density_error'] + 1e3 * relu(q['beltrami_max'] - 1)
    fc, fd = loss_at(c), loss_at(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - (b - a) / gr
            fc = loss_at(c)
        else:
            a, fc = c, fd
            c = d
            d = a + (b - a) / gr
            fd = loss_at(d)
    alpha_opt = (a + b) / 2
    loss_opt  = loss_at(alpha_opt)
    return alpha_opt, loss_opt

def optimize_refinement_weight_ft(model, ft_model,
                               X, Y, p, faces, dx,
                               lb=0.1, ub=3.0, tol=1e-4, max_iter=50) -> tuple:
    u_init, v_init = model(torch.stack([X, Y, p], dim=0).unsqueeze(0))[0]
    u_init = X + X * (1 - X) * u_init
    v_init = Y + Y * (1 - Y) * v_init
    ft_u_init, ft_v_init = ft_model(torch.stack([u_init, v_init, p], dim=0).unsqueeze(0))[0]
    ft_u_init = u_init + u_init * (1 - u_init) * ft_u_init
    ft_v_init = v_init + v_init * (1 - v_init) * ft_v_init
    m_init = compute_mapping_quality(X, Y, p, ft_u_init, ft_v_init, faces, dx)

    a = lb
    b = ub
    
    if (m_init['beltrami_max'] >= 1):
        b = 1.0  # If initial mapping is already bad, start with a lower b
    
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    def loss_at(alpha):
        u, v = model(torch.stack([X, Y, p], dim=0).unsqueeze(0))[0]
        u = X + X * (1 - X) * u * alpha
        v = Y + Y * (1 - Y) * v * alpha
        ft_u, ft_v = ft_model(torch.stack([u, v, p], dim=0).unsqueeze(0))[0]
        ft_u = u + u * (1 - u) * ft_u
        ft_v = v + v * (1 - v) * ft_v
        m = compute_mapping_quality(X, Y, p, ft_u, ft_v, faces, dx)
        return m['density_error'] + 1e3*relu(m['beltrami_max'] - 1)

    fc, fd = loss_at(c), loss_at(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - (b - a) / gr
            fc = loss_at(c)
        else:
            a, fc = c, fd
            c = d
            d = a + (b - a) / gr
            fd = loss_at(d)

    alpha_opt = (a + b) / 2
    loss_opt = loss_at(alpha_opt)
    return alpha_opt, loss_opt
