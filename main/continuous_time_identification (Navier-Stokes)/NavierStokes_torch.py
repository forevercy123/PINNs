import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import time

# 设置随机种子以保证可重复性
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# 设备选择（CPU or GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PhysicsInformedNN(nn.Module):
    def __init__(self, x, y, t, u, v, layers):
        super().__init__()
        X = np.concatenate([x, y, t], 1)
        self.lb = torch.tensor(X.min(0), dtype=torch.float32).to(device)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32).to(device)

        # Convert to tensors and move to device
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Learnable parameters
        self.lambda_1 = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32).to(device)
        )
        self.lambda_2 = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32).to(device)
        )

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            in_dim = layers[l]
            out_dim = layers[l + 1]
            W = nn.Parameter(torch.empty(in_dim, out_dim, device=device))
            b = nn.Parameter(torch.zeros(out_dim, device=device))
            nn.init.xavier_normal_(W)
            weights.append(W)
            biases.append(b)
        return nn.ParameterList(weights), nn.ParameterList(biases)

    def neural_net(self, X):
        H = (X - self.lb) / (self.ub - self.lb)
        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.tanh(H @ W + b)
        W = self.weights[-1]
        b = self.biases[-1]
        Y = H @ W + b
        return Y

    def net_NS(self, x, y, t):
        X = torch.cat([x, y, t], dim=1)
        psi_and_p = self.neural_net(X)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = torch.autograd.grad(
            psi, y, grad_outputs=torch.ones_like(psi), create_graph=True
        )[0]
        v = -torch.autograd.grad(
            psi, x, grad_outputs=torch.ones_like(psi), create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        v_x = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v), create_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(v), create_graph=True
        )[0]
        v_t = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v), create_graph=True
        )[0]

        p_x = torch.autograd.grad(
            p, x, grad_outputs=torch.ones_like(p), create_graph=True
        )[0]
        p_y = torch.autograd.grad(
            p, y, grad_outputs=torch.ones_like(p), create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True
        )[0]

        f_u = (
            u_t
            + self.lambda_1 * (u * u_x + v * u_y)
            + p_x
            - self.lambda_2 * (u_xx + u_yy)
        )
        f_v = (
            v_t
            + self.lambda_1 * (u * v_x + v * v_y)
            + p_y
            - self.lambda_2 * (v_xx + v_yy)
        )

        return u, v, p, f_u, f_v

    def loss_fn(self):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_v = torch.mean((self.v - v_pred) ** 2)
        loss_f_u = torch.mean(f_u_pred**2)
        loss_f_v = torch.mean(f_v_pred**2)
        loss = loss_u + loss_v + loss_f_u + loss_f_v
        return loss

    def train_Adam(self, nIter):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        print("Training with Adam...")
        start_time = time.time()
        for it in range(nIter):
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            optimizer.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print(f"It: {it}, Loss: {loss.item():.3e}, Time: {elapsed:.2f}s")
                start_time = time.time()

    def closure(self):
        self.optimizer_LBFGS.zero_grad()
        loss = self.loss_fn()
        loss.backward()
        return loss

    def train_LBFGS(self, nIter):
        print("Training with L-BFGS...")
        self.optimizer_LBFGS = optim.LBFGS(
            self.parameters(),
            max_iter=50,
            max_eval=50,
            tolerance_change=1e-8,
            line_search_fn="strong_wolfe",
        )
        for it in range(nIter // 50):  # Total iterations ~50k
            self.optimizer_LBFGS.step(self.closure)
            loss = self.loss_fn().item()
            lambda_1 = self.lambda_1.item()
            lambda_2 = self.lambda_2.item()
            print(
                f"Iteration {it * 50}: Loss={loss:.3e}, λ1={lambda_1:.3f}, λ2={lambda_2:.5f}"
            )

    def predict(self, x_star, y_star, t_star):
        x = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
        y = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
        t = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)
        # 确保 t 是二维张量
        if t.dim() == 1:
            t = t.unsqueeze(1)  # 从 (N,) 重塑为 (N, 1)
        u, v, p, _, _ = self.net_NS(x, y, t)
        return (
            u.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )


if __name__ == "__main__":
    N_train = 5000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    data = loadmat("../Data/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2

    N, T = X_star.shape[0], t_star.shape[0]
    print(f"t_star shape: {t_star.shape}, T: {T}")  # 调试：打印 t_star 形状

    # 确保 t_star 是 (T, 1) 形状
    t_star = t_star.reshape(-1, 1)
    print(f"t_star reshaped: {t_star.shape}")

    # 修正 TT 的生成，确保形状为 (N, T)
    TT = np.tile(t_star.T, (N, 1))  # 先转置 t_star 为 (1, T)，然后复制 N 次
    print(f"TT shape: {TT.shape}")  # 调试：打印 TT 形状

    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))

    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]

    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]

    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers).to(
        device
    )
    model.train_Adam(1000)
    model.train_LBFGS(5000)

    # 动态选择 snap
    print(f"Number of time steps (T): {T}")  # 打印时间步数以供调试
    if T > 1:
        snap = np.array([min(100, T - 1)])  # 选择最后一个时间步或100（取较小值）
    else:
        snap = np.array([0])  # 如果只有1个时间步，选择索引0
    print(f"snap value: {snap}, snap shape: {snap.shape}")  # 调试：打印 snap 值和形状

    # 确保 snap 索引合法
    if snap[0] >= TT.shape[1]:
        raise ValueError(
            f"snap index {snap[0]} out of bounds for TT with shape {TT.shape}"
        )

    x_star = X_star[:, 0:1]  # (N, 1)
    y_star = X_star[:, 1:2]  # (N, 1)
    t_star = TT[:, snap[0]][:, None]  # 确保 t_star 是 (N, 1)

    print(
        f"x_star shape: {x_star.shape}, y_star shape: {y_star.shape}, t_star shape: {t_star.shape}"
    )  # 调试：打印输入形状

    u_star = U_star[:, 0, snap[0]][:, None]
    v_star = U_star[:, 1, snap[0]][:, None]
    p_star = P_star[:, snap[0]][:, None]

    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

    lambda_1_value = model.lambda_1.item()
    lambda_2_value = model.lambda_2.item()

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100

    print(f"Error u: {error_u:.2e}")
    print(f"Error v: {error_v:.2e}")
    print(f"Error p: {error_p:.2e}")
    print(f"Error l1: {error_lambda_1:.5f}%")
    print(f"Error l2: {error_lambda_2:.5f}%")

    # Plot Results
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x_vals = np.linspace(lb[0], ub[0], nn)
    y_vals = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x_vals, y_vals)

    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method="cubic")
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method="cubic")
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method="cubic")
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method="cubic")
    U_exact = griddata(X_star, u_star.flatten(), (X, Y), method="cubic")
    V_exact = griddata(X_star, v_star.flatten(), (X, Y), method="cubic")

    # 可视化部分
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    # Predicted u
    c = ax[0, 0].contourf(X, Y, UU_star, levels=50, cmap="jet")
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_title("Predicted u")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("y")
    # Predicted v
    c = ax[0, 1].contourf(X, Y, VV_star, levels=50, cmap="jet")
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_title("Predicted v")
    ax[0, 1].set_xlabel("x")
    ax[0, 1].set_ylabel("y")
    # Predicted p
    c = ax[0, 2].contourf(X, Y, PP_star, levels=50, cmap="jet")
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_title("Predicted p")
    ax[0, 2].set_xlabel("x")
    ax[0, 2].set_ylabel("y")
    # Exact u
    c = ax[1, 0].contourf(X, Y, U_exact, levels=50, cmap="jet")
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_title("Exact u")
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("y")
    # Exact v
    c = ax[1, 1].contourf(X, Y, V_exact, levels=50, cmap="jet")
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_title("Exact v")
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("y")
    # Exact p
    c = ax[1, 2].contourf(X, Y, P_exact, levels=50, cmap="jet")
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_title("Exact p")
    ax[1, 2].set_xlabel("x")
    ax[1, 2].set_ylabel("y")
    plt.tight_layout()
    plt.show()
