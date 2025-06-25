"""
@author: Maziar Raissi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from Utilities.plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置随机种子以确保可重复性
np.random.seed(1234)
torch.manual_seed(1234)

# 设备配置：优先使用GPU，如果没有GPU则回退到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PhysicsInformedNN:
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        """
        初始化PhysicsInformedNN类
        参数:
            x0: 初始条件的位置
            u0, v0: 初始条件的实部和虚部
            tb: 时间边界点
            X_f: 配点法点
            layers: 神经网络层结构
            lb, ub: 域边界
        """
        # 数据准备
        X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.x0 = torch.tensor(X0[:, 0:1], dtype=torch.float32, requires_grad=True).to(
            device
        )
        self.t0 = torch.tensor(X0[:, 1:2], dtype=torch.float32, requires_grad=True).to(
            device
        )

        self.x_lb = torch.tensor(
            X_lb[:, 0:1], dtype=torch.float32, requires_grad=True
        ).to(device)
        self.t_lb = torch.tensor(
            X_lb[:, 1:2], dtype=torch.float32, requires_grad=True
        ).to(device)

        self.x_ub = torch.tensor(
            X_ub[:, 0:1], dtype=torch.float32, requires_grad=True
        ).to(device)
        self.t_ub = torch.tensor(
            X_ub[:, 1:2], dtype=torch.float32, requires_grad=True
        ).to(device)

        self.x_f = torch.tensor(
            X_f[:, 0:1], dtype=torch.float32, requires_grad=True
        ).to(device)
        self.t_f = torch.tensor(
            X_f[:, 1:2], dtype=torch.float32, requires_grad=True
        ).to(device)

        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.v0 = torch.tensor(v0, dtype=torch.float32).to(device)

        # 初始化神经网络
        self.layers = layers
        self.model = self.initialize_NN(layers).to(device)

        # 优化器
        self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # L-BFGS优化器
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=1000,
            max_eval=1000,
            history_size=50,
            tolerance_grad=1.0 * np.finfo(float).eps,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

    def initialize_NN(self, layers):
        """
        初始化神经网络
        """
        modules = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # 非输出层添加激活函数
                modules.append(torch.nn.Tanh())
        model = torch.nn.Sequential(*modules)

        # Xavier 初始化
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)

        return model

    def neural_net(self, X):
        """
        前向传播
        """
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(H)

    def net_uv(self, x, t):
        """
        计算 u, v 及其导数
        """
        X = torch.cat([x, t], dim=1)
        uv = self.neural_net(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u).to(device), create_graph=True
        )[0]
        v_x = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v).to(device), create_graph=True
        )[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        """
        计算PDE残差
        """
        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u).to(device), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x).to(device), create_graph=True
        )[0]

        v_t = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v).to(device), create_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x, x, grad_outputs=torch.ones_like(v_x).to(device), create_graph=True
        )[0]

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v

    def loss_fn(self):
        """
        计算总损失
        """
        u0_pred, v0_pred, _, _ = self.net_uv(self.x0, self.t0)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(
            self.x_lb, self.t_lb
        )
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(
            self.x_ub, self.t_ub
        )
        f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)

        loss = (
            torch.mean(torch.square(self.u0 - u0_pred))
            + torch.mean(torch.square(self.v0 - v0_pred))
            + torch.mean(torch.square(u_lb_pred - u_ub_pred))
            + torch.mean(torch.square(v_lb_pred - v_ub_pred))
            + torch.mean(torch.square(u_x_lb_pred - u_x_ub_pred))
            + torch.mean(torch.square(v_x_lb_pred - v_x_ub_pred))
            + torch.mean(torch.square(f_u_pred))
            + torch.mean(torch.square(f_v_pred))
        )

        return loss

    def train(self, nIter):
        """
        训练模型
        """
        start_time = time.time()
        for it in range(nIter):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.optimizer_Adam.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print(f"It: {it}, Loss: {loss.item():.3e}, Time: {elapsed:.2f}")
                start_time = time.time()

        # L-BFGS优化
        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            print(f"Loss: {loss.item()}")
            return loss

        self.optimizer_LBFGS.step(closure)

    def predict(self, X_star):
        """
        预测
        """
        self.model.eval()
        # 确保 X_star 有效
        assert not np.any(np.isnan(X_star)), "X_star contains NaN"
        assert not np.any(np.isinf(X_star)), "X_star contains Inf"

        x_star = torch.tensor(
            X_star[:, 0:1], dtype=torch.float32, requires_grad=True
        ).to(device)
        t_star = torch.tensor(
            X_star[:, 1:2], dtype=torch.float32, requires_grad=True
        ).to(device)

        u_star, v_star, _, _ = self.net_uv(x_star, t_star)
        f_u_star, f_v_star = self.net_f_uv(x_star, t_star)

        return (
            u_star.detach().cpu().numpy(),
            v_star.detach().cpu().numpy(),
            f_u_star.detach().cpu().numpy(),
            f_v_star.detach().cpu().numpy(),
        )


if __name__ == "__main__":
    noise = 0.0
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]

    data = scipy.io.loadmat("../Data/NLS.mat")

    t = data["tt"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = data["uu"]
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    start_time = time.time()
    model.train(1000)
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.4f}")

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    print(f"u_pred shape: {u_pred.shape}, v_pred shape: {v_pred.shape}")
    print(f"u_pred sample: {u_pred[:5].flatten()}")
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print(f"Error u: {error_u:.6e}")
    print(f"Error v: {error_v:.6e}")
    print(f"Error h: {error_h:.6e}")

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method="cubic")
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method="cubic")

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method="cubic")
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method="cubic")

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    # 创建新图形
    fig = plt.figure(figsize=(10, 9))  # 使用 plt.figure 替代 newfig
    ax = fig.add_subplot(111)  # 显式创建主轴
    ax.axis("off")

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)

    # 移除现有轴（如果有）
    if ax in fig.axes:
        ax.remove()
    ax = fig.add_subplot(gs0[:, :])

    h = ax.imshow(
        H_pred.T,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, "k--", linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, "k--", linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, "k--", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$|h(t,x)|$", fontsize=10)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax1 = fig.add_subplot(gs1[0, 0])
    ax1.plot(x, Exact_h[:, 75], "b-", linewidth=2, label="Exact")
    ax1.plot(x, H_pred[75, :], "r--", linewidth=2, label="Prediction")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$|h(t,x)|$")
    ax1.set_title("$t = %.2f$" % float(t[75]), fontsize=10)
    ax1.axis("square")
    ax1.set_xlim([-5.1, 5.1])
    ax1.set_ylim([-0.1, 5.1])

    ax2 = fig.add_subplot(gs1[0, 1])
    ax2.plot(x, Exact_h[:, 100], "b-", linewidth=2, label="Exact")
    ax2.plot(x, H_pred[100, :], "r--", linewidth=2, label="Prediction")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$|h(t,x)|$")
    ax2.axis("square")
    ax2.set_xlim([-5.1, 5.1])
    ax2.set_ylim([-0.1, 5.1])
    ax2.set_title("$t = %.2f$" % float(t[100]), fontsize=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax3 = fig.add_subplot(gs1[0, 2])
    ax3.plot(x, Exact_h[:, 125], "b-", linewidth=2, label="Exact")
    ax3.plot(x, H_pred[125, :], "r--", linewidth=2, label="Prediction")
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$|h(t,x)|$")
    ax3.axis("square")
    ax3.set_xlim([-5.1, 5.1])
    ax3.set_ylim([-0.1, 5.1])
    ax3.set_title("$t = %.2f$" % float(t[125]), fontsize=10)

    plt.tight_layout()
    plt.show()

    # savefig('./figures/NLS')
