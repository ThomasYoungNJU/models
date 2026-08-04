import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



WindowProcess = None
WindowProcessReverse = None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: tuple):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size(Wt, Wh, Ww)
    Returns:
        windows: (num_windows*B, window_size, C)
    """
    if len(window_size) == 3:
        B, T, H, W, C = x.shape
        x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
        # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
        # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    elif len(window_size) == 2:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
        # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows



def window_reverse(windows, window_size, T=1, H=1, W=1):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    if len(window_size) == 3:
        B = int(windows.shape[0] / (T * H * W / window_size[0] / window_size[1] / window_size[2]))
        # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
        x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
        # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
        # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    elif len(window_size) == 2:
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
        # print(H, W, window_size, flush=True)
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
        # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class rope2(nn.Module):
    def __init__(self, shape, dim, origin_shape=[0,0]) -> None:
        super().__init__()

        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])

        if origin_shape[0] > 0:
            coords_0 = coords_0 / (shape[0] - 1) * (origin_shape[0] - 1)
            coords_1 = coords_1 / (shape[1] - 1) * (origin_shape[1] - 1)
        coords = torch.stack(torch.meshgrid([coords_0, coords_1], indexing="ij")).reshape(2, -1)

        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 10000 ** -freq_seq1
        inv_freq2 = 10000 ** -freq_seq2

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x)
        self.cos1 = self.cos1.to(x)
        self.sin2 = self.sin2.to(x)
        self.cos2 = self.cos2.to(x)

        x11, x21, x12, x22 = x.split([self.dim1_size, self.dim2_size, \
                                        self.dim1_size, self.dim2_size], dim=-1)

        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)

        return res





class SD_attn(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., shift_size=[0, 0, 0], dilated_size=[1,1,1]) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = torch.tensor(head_dim ** -0.5)

        self.dilated_size = dilated_size[-len(window_size):]
        self.window_size = window_size
        self.shift_size = shift_size
        self.total_window_size = [window_size[i] * dilated_size[i] for i in range(len(window_size))]



        self.rope_quad = rope2(self.window_size, head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.position_enc = rope2(window_size, head_dim)


    def create_mask(self, x):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        # Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        # Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition


        if len(self.window_size) == 3:
            _, T, H, W, _ = x.shape
            img_mask = torch.zeros((1, T, H, W, 1), device=x.device)  # [1, Hp, Wp, 1]
            t_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], 0),
                        slice(0, None))
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1
        elif len(self.window_size) == 2:
            _, H, W, _ = x.shape
            img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, Hp, Wp, 1]
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], 0),
                        slice(0, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.total_window_size)  # [B, nW, Mt, Mh, Mw, C]
        mask_windows = mask_windows.reshape(-1, *self.total_window_size, 1)
        B_ = mask_windows.shape[0]
        if len(self.dilated_size) == 3:
            mask_windows = window_partition(mask_windows, self.dilated_size).reshape(B_, -1,
                                        self.dilated_size[0]*self.dilated_size[1]*self.dilated_size[2], 1).permute(
                                        0, 2, 1, 3).reshape(B_*self.dilated_size[0]*self.dilated_size[1]*self.dilated_size[2], -1)
        elif len(self.dilated_size) == 2:
            mask_windows = window_partition(mask_windows, self.dilated_size).reshape(B_, -1,
                                        self.dilated_size[0]*self.dilated_size[1], 1).permute(
                                        0, 2, 1, 3).reshape(B_*self.dilated_size[0]*self.dilated_size[1], -1)


        # mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        # if len(self.window_size) == 3:
        #     mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])  # [nW, Mh*Mw]
        # elif len(self.window_size) == 2:
        #     mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])  # [nW, Mh*Mw]

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -torch.inf).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size, Mt, Mh, Mw, total_embed_dim]
        T=1

        if len(self.window_size) == 2:
            _, H, W, C = x.shape
        elif len(self.window_size) == 3:
            _, T, H, W, C = x.shape

        if (self.shift_size[-1] == 0) or (self.total_window_size[-1] == W):
            mask = None
        else:
            mask = self.create_mask(x).to(x)

        if self.shift_size[-1] > 0:
            if len(self.window_size) == 3:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            elif len(self.window_size) == 2:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x=x
            mask = None

        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]

        x_windows = window_partition(shifted_x, self.total_window_size)  # [B, nW, Mt, Mh, Mw, C]
        x_windows = x_windows.reshape(-1, *self.total_window_size, C)
        B = x_windows.shape[0]
        if len(self.dilated_size) == 3:
            x_windows = window_partition(x_windows, self.dilated_size).reshape(B, -1,
                                        self.dilated_size[0]*self.dilated_size[1]*self.dilated_size[2], C).permute(
                                        0, 2, 1, 3).reshape(B*self.dilated_size[0]*self.dilated_size[1]*self.dilated_size[2], -1, C)
        elif len(self.dilated_size) == 2:
            x_windows = window_partition(x_windows, self.dilated_size).reshape(B, -1,
                                        self.dilated_size[0]*self.dilated_size[1], C).permute(
                                        0, 2, 1, 3).reshape(B*self.dilated_size[0]*self.dilated_size[1], -1, C)
        B_, N, C = x_windows.shape


        qkv = self.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = self.position_enc(q.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)
        k = self.position_enc(k.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale.to(q)
        attn = (q @ k.transpose(-2, -1))


        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]

        save_attn = attn.mean(dim=-3)

        attn_windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        if len(self.window_size) == 3:
            attn_windows = attn_windows.reshape(B, -1, N, C).permute(0, 2, 1, 3).reshape(
                                            -1, self.dilated_size[0]*self.dilated_size[1]*self.dilated_size[2], C)
            attn_windows = window_reverse(attn_windows, self.dilated_size, *self.total_window_size)
        elif len(self.window_size) == 2:
            attn_windows = attn_windows.reshape(B, -1, N, C).permute(0, 2, 1, 3).reshape(
                                            -1, self.dilated_size[0]*self.dilated_size[1], C)
            attn_windows = window_reverse(attn_windows, self.dilated_size, 1, *self.total_window_size)

        shifted_x = window_reverse(attn_windows, self.total_window_size, T, H, W)

        if self.shift_size[0] > 0:
            if len(self.window_size) == 3:
                x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            elif len(self.window_size) == 2:
                x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, save_attn



class Windowattn_block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4.,
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                attn_type="windowattn", pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm
        self.attn_type = attn_type
        if "save_attn" in kwargs:
            self.save_attn = kwargs['save_attn']
        else:
            self.save_attn = False

        self.norm = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')
        if attn_type == "windowattn":
            if "shift_size" not in kwargs:
                shift_size = [0, 0, 0]
            else:
                shift_size = kwargs["shift_size"]
            if "dilated_size" in kwargs:
                dilated_size = kwargs["dilated_size"]
            else:
                dilated_size = [1, 1, 1]
            self.attn = SD_attn(
                dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop, shift_size=shift_size, dilated_size=dilated_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)




    def forward(self, x):
        shortcut = x
        # partition windows

        if self.pre_norm:
            if self.attn_type == "windowattn":
                x, save_attn = self.attn(self.norm(x))
            else:
                x = self.attn(self.norm(x))

            x = shortcut + self.drop_path(x)

            # x = shortcut + self.drop_path(self.attn(self.norm(x)))
        else:
            if self.attn_type == "windowattn":
                x, save_attn = self.attn(x)
            else:
                x = self.attn(x)

            x = self.norm(shortcut + self.drop_path(x))

            # x = self.norm(shortcut + self.drop_path(self.attn(x)))

        # W-MSA/SW-MS

        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))

        if self.attn_type == "windowattn" and self.save_attn:
            return x, save_attn
        else:
            return x


