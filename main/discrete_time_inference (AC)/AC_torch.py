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

# 初始化 CUDA 上下文
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    print(
        f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


class PhysicsInformedNN:
    def __init__(
        self, x0, u0, x1, layers, dt, lb, ub, q, max_iter_num=50000, max_eval_num=50000
    ):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(device)
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)

        self.layers = layers
        self.dt = dt
        self.q = max(q, 1)

        # 初始化神经网络
        self.model = self.initialize_NN(layers).to(device)

        # 加载IRK权重
        tmp = np.float32(
            np.loadtxt("../../Utilities/IRK_weights/Butcher_IRK%d.txt" % (q), ndmin=2)
        )
        weights = np.reshape(tmp[0 : q**2 + q], (q + 1, q))
        self.IRK_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.IRK_times = torch.tensor(tmp[q**2 + q :], dtype=torch.float32).to(device)

        # 优化器
        self.optimizer_Adam = torch.optim.Adam(list(self.model.parameters()), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam, step_size=10000, gamma=0.5
        )

        self.optimizer_LBFGS = torch.optim.LBFGS(
            list(self.model.parameters()),
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
        U1 = self.neural_net(x)
        U = U1[:, :-1]
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_xx
        U0 = U1 - self.dt * torch.matmul(F, self.IRK_weights.T)
        return U0

    def net_U1(self, x):
        U1 = self.neural_net(x)
        U1_x = self.fwd_gradients(U1, x)
        return U1, U1_x

    def loss_fn(self):
        U0_pred = self.net_U0(self.x0)
        U1_pred, U1_x_pred = self.net_U1(self.x1)
        loss = (
            torch.mean(torch.square(self.u0 - U0_pred))
            + torch.mean(torch.square(U1_pred[0, :] - U1_pred[1, :]))
            + torch.mean(torch.square(U1_x_pred[0, :] - U1_x_pred[1, :]))
        )
        return loss

    def train(self, nIter):
        start_time = time.time()
        for it in range(nIter):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.optimizer_Adam.step()
            self.scheduler.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print(f"It: {it}, Loss: {loss.item():.3e}, Time: {elapsed:.2f}")
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
        U1_star, _ = self.net_U1(x_star_torch)
        return U1_star.detach().cpu().numpy()


if __name__ == "__main__":
    q = 100
    layers = [1, 300, 300, 300, 300, q + 1]
    lb = np.array([-1.0])
    ub = np.array([1.0])

    N = 200

    data = scipy.io.loadmat("../Data/AC.mat")

    t = data["tt"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["uu"]).T

    idx_t0 = 20
    idx_t1 = 180
    dt = float(t[idx_t1] - t[idx_t0])

    # Initial data
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_t0 : idx_t0 + 1, idx_x].T
    u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    # Boundary data
    x1 = np.vstack((lb, ub))

    # Test data
    x_star = x

    model = PhysicsInformedNN(
        x0, u0, x1, layers, dt, lb, ub, q, max_iter_num=50000, max_eval_num=50000
    )
    model.train(nIter=50000)

    U1_pred = model.predict(x_star)

    error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(
        Exact[idx_t1, :], 2
    )
    print(f"Error: {error:.6e}")

    # Plotting
    fig = plt.figure(figsize=(10, 12))
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.95, bottom=0.65, left=0.15, right=0.85, wspace=0)

    ax = fig.add_subplot(gs0[:, :])
    h = ax.imshow(
        Exact.T,
        interpolation="nearest",
        cmap="seismic",
        extent=[t.min(), t.max(), x.min(), x.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[idx_t0] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(t[idx_t1] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("$u(t,x)$", fontsize=10)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.55, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax = fig.add_subplot(gs1[0, 0])
    ax.plot(x, Exact[idx_t0, :], "b-", linewidth=2, label="Exact")
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(f"$t = {float(t[idx_t0]):.2f}$\n{N} training data", fontsize=10)
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax = fig.add_subplot(gs1[0, 1])
    ax.plot(x, Exact[idx_t1, :], "b-", linewidth=2, label="Exact")
    ax.plot(x_star, U1_pred[:, -1], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(f"$t = {float(t[idx_t1]):.2f}$", fontsize=10)
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

    plt.subplots_adjust(
        top=0.95, bottom=0.15, left=0.15, right=0.85, hspace=0.4, wspace=0.5
    )
    plt.savefig("AC_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
