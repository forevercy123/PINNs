"""
@author: Maziar Raissi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置随机种子以确保可重复性
np.random.seed(1234)
torch.manual_seed(1234)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PhysicsInformedNN:
    def __init__(
        self,
        x0,
        u0,
        x1,
        u1,
        layers,
        dt,
        lb,
        ub,
        q,
        max_iter_num=50000,
        max_eval_num=10000,
    ):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(device)

        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.u1 = torch.tensor(u1, dtype=torch.float32).to(device)

        self.layers = layers
        self.dt = dt
        self.q = max(q, 1)

        # 初始化神经网络
        self.model = self.initialize_NN(layers).to(device)

        # 初始化参数
        self.lambda_1 = torch.nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32).to(device)
        )
        self.lambda_2 = torch.nn.Parameter(
            torch.tensor([-6.0], dtype=torch.float32).to(device)
        )

        # 加载IRK权重
        tmp = np.float32(
            np.loadtxt("../../Utilities/IRK_weights/Butcher_IRK%d.txt" % (q), ndmin=2)
        )
        weights = np.reshape(tmp[0 : q**2 + q], (q + 1, q))
        self.IRK_alpha = torch.tensor(weights[0:-1, :], dtype=torch.float32).to(device)
        self.IRK_beta = torch.tensor(weights[-1:, :], dtype=torch.float32).to(device)
        self.IRK_times = torch.tensor(tmp[q**2 + q :], dtype=torch.float32).to(device)

        # 优化器
        self.optimizer_Adam = torch.optim.Adam(
            [self.lambda_1, self.lambda_2] + list(self.model.parameters()), lr=0.0001
        )

        self.optimizer_LBFGS = torch.optim.LBFGS(
            [self.lambda_1, self.lambda_2] + list(self.model.parameters()),
            lr=1.0,
            max_iter=max_iter_num,
            max_eval=max_eval_num,
            history_size=50,
            tolerance_grad=1.0 * np.finfo(float).eps,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

    def initialize_NN(self, layers):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(torch.nn.Tanh())
        model = torch.nn.Sequential(*modules)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)
        return model

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(H)

    def fwd_gradients(self, U, x):
        g = torch.autograd.grad(
            U, x, grad_outputs=torch.ones_like(U).to(device), create_graph=True
        )[0]
        return g

    def net_U0(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = torch.clamp(torch.exp(self.lambda_2), min=1e-6, max=1.0)
        U = self.neural_net(x)
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        U_xxx = self.fwd_gradients(U_xx, x)
        F = -lambda_1 * U * U_x - lambda_2 * U_xxx
        U0 = U - self.dt * torch.matmul(F, self.IRK_alpha.T)
        return U0

    def net_U1(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = torch.clamp(torch.exp(self.lambda_2), min=1e-6, max=1.0)
        U = self.neural_net(x)
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        U_xxx = self.fwd_gradients(U_xx, x)
        F = -lambda_1 * U * U_x - lambda_2 * U_xxx
        U1 = U + self.dt * torch.matmul(F, (self.IRK_beta - self.IRK_alpha).T)
        return U1

    def loss_fn(self):
        U0_pred = self.net_U0(self.x0)
        U1_pred = self.net_U1(self.x1)
        loss = torch.sum(torch.square(self.u0 - U0_pred)) + torch.sum(
            torch.square(self.u1 - U1_pred)
        )
        return loss

    def train(self, nIter):
        start_time = time.time()
        for it in range(nIter):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.optimizer_Adam.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.item()
                lambda_2_value = np.exp(self.lambda_2.item())
                print(
                    f"It: {it}, Loss: {loss.item():.3e}, l1: {lambda_1_value:.3f}, l2: {lambda_2_value:.5f}, Time: {elapsed:.2f}"
                )
                start_time = time.time()

        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            print(f"Loss: {loss.item()}")
            return loss

        self.optimizer_LBFGS.step(closure)

    def predict(self, x_star):
        self.model.eval()
        assert not np.any(np.isnan(x_star)), "x_star contains NaN"
        assert not np.any(np.isinf(x_star)), "x_star contains Inf"

        x_star_torch = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(
            device
        )
        U0_star = self.net_U0(x_star_torch)
        U1_star = self.net_U1(x_star_torch)

        return U0_star.detach().cpu().numpy(), U1_star.detach().cpu().numpy()


if __name__ == "__main__":
    q = 50
    skip = 120

    N0 = 199
    N1 = 201
    layers = [1, 50, 50, 50, 50, q]

    data = scipy.io.loadmat("../Data/KdV.mat")

    t_star = data["tt"].flatten()[:, None]
    x_star = data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])

    idx_t = 40

    # Noiseless Data
    noise = 0.0
    idx_x0 = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x0, :]
    u0 = Exact[idx_x0, idx_t][:, None]
    u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    idx_x1 = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x1, :]
    u1 = Exact[idx_x1, idx_t + skip][:, None]
    u1 = u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])

    dt = float(t_star[idx_t + skip] - t_star[idx_t])

    lb = x_star.min(0)
    ub = x_star.max(0)

    model = PhysicsInformedNN(
        x0, u0, x1, u1, layers, dt, lb, ub, q, max_iter_num=500, max_eval_num=100
    )
    model.train(nIter=500)

    torch.cuda.empty_cache()
    U0_pred, U1_pred = model.predict(x_star)

    lambda_1_value = model.lambda_1.item()
    lambda_2_value = np.exp(model.lambda_2.item())

    error_lambda_1 = np.abs(lambda_1_value - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.0025) / 0.0025 * 100

    print(f"Error lambda_1: {error_lambda_1:.6f}%")
    print(f"Error lambda_2: {error_lambda_2:.6f}%")

    # Noisy Data
    noise = 0.01
    idx_x0 = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x0, :]
    u0 = Exact[idx_x0, idx_t][:, None]
    u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    idx_x1 = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x1, :]
    u1 = Exact[idx_x1, idx_t + skip][:, None]
    u1 = u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])

    model = PhysicsInformedNN(
        x0, u0, x1, u1, layers, dt, lb, ub, q, max_iter_num=500, max_eval_num=100
    )
    model.train(nIter=500)

    torch.cuda.empty_cache()
    U0_pred, U1_pred = model.predict(x_star)

    lambda_1_value_noisy = model.lambda_1.item()
    lambda_2_value_noisy = np.exp(model.lambda_2.item())

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) / 1.0 * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.0025) / 0.0025 * 100

    print(f"Error lambda_1: {error_lambda_1_noisy:.6f}%")
    print(f"Error lambda_2: {error_lambda_2_noisy:.6f}%")

    # Plotting
    fig = plt.figure(figsize=(10, 15))
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.95, bottom=0.65, left=0.1, right=0.9, wspace=0.05)

    ax = fig.add_subplot(gs0[:, :])
    h = ax.imshow(
        Exact,
        interpolation="nearest",
        cmap="rainbow",
        extent=[t_star.min(), t_star.max(), lb[0], ub[0]],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(x_star.min(), x_star.max(), 2)[:, None]
    ax.plot(t_star[idx_t] * np.ones((2, 1)), line, "w-", linewidth=1.0)
    ax.plot(t_star[idx_t + skip] * np.ones((2, 1)), line, "w-", linewidth=1.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("$u(t,x)$", fontsize=10)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.55, bottom=0.25, left=0.1, right=0.9, wspace=0.5)

    ax1 = fig.add_subplot(gs1[0, 0])
    ax1.plot(x_star, Exact[:, idx_t][:, None], "b", linewidth=2, label="Exact")
    ax1.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u(t,x)$")
    ax1.set_title(
        "$t = %.2f$\n%d training data" % (float(t_star[idx_t]), u0.shape[0]),
        fontsize=10,
    )

    ax2 = fig.add_subplot(gs1[0, 1])
    ax2.plot(x_star, Exact[:, idx_t + skip][:, None], "b", linewidth=2, label="Exact")
    ax2.plot(x1, u1, "rx", linewidth=2, label="Data")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$u(t,x)$")
    ax2.set_title(
        "$t = %.2f$\n%d training data" % (float(t_star[idx_t + skip]), u1.shape[0]),
        fontsize=10,
    )
    ax2.legend(loc="upper center", bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.15, bottom=0.05, left=0.1, right=0.9, wspace=0.0)

    ax3 = fig.add_subplot(gs2[0, 0])
    ax3.axis("off")
    s1 = r"Correct PDE: $u_t + u u_x + 0.0025 u_{xxx} = 0$"
    s2 = r"Identified PDE (clean): $u_t + %.3f u u_x + %.7f u_{xxx} = 0$" % (
        lambda_1_value,
        abs(lambda_2_value),
    )
    s3 = r"Identified PDE (1%% noise): $u_t + %.3f u u_x + %.7f u_{xxx} = 0$" % (
        lambda_1_value_noisy,
        abs(lambda_2_value_noisy),
    )
    s = "\n".join([s1, s2, s3])
    ax3.text(-0.1, 0.3, s, fontsize=10, verticalalignment="top")

    plt.subplots_adjust(
        top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=0.4, wspace=0.5
    )
    plt.show()
