import sys

sys.path.insert(0, "../../Utilities/")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata
from scipy.optimize import minimize
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from Utilities.plotting import newfig, savefig
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.random.set_seed(1234)


class PhysicsInformedNN(tf.Module):
    def __init__(self, x, y, t, u, v, layers):
        super().__init__()
        X = np.concatenate([x, y, t], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)

        # Convert to tensors
        self.x = tf.constant(x, dtype=tf.float32)
        self.y = tf.constant(y, dtype=tf.float32)
        self.t = tf.constant(t, dtype=tf.float32)
        self.u = tf.constant(u, dtype=tf.float32)
        self.v = tf.constant(v, dtype=tf.float32)

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32, name="lambda_1")
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32, name="lambda_2")

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            in_dim = layers[l]
            out_dim = layers[l + 1]
            W = self.xavier_init([in_dim, out_dim])
            b = tf.Variable(tf.zeros([1, out_dim], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, shape):
        in_dim, out_dim = shape
        stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(shape, stddev=stddev), dtype=tf.float32
        )

    def neural_net(self, X, weights, biases):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(weights) - 1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            # 所有操作都必须在这个 tape 上下文中进行
            tape.watch([x, y, t])

            # 前向传播：神经网络输出 psi 和 p
            psi_and_p = self.neural_net(
                tf.concat([x, y, t], axis=1), self.weights, self.biases
            )
            psi = psi_and_p[:, 0:1]
            p = psi_and_p[:, 1:2]

            # 计算 u, v
            u = tape.gradient(psi, y)
            v = -tape.gradient(psi, x)

            # 一阶导数
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_t = tape.gradient(u, t)

            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_t = tape.gradient(v, t)

            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)

            # 二阶导数
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            v_xx = tape.gradient(v_x, x)
            v_yy = tape.gradient(v_y, y)

        # 删除 tape
        del tape

        # 构建 PDE 残差
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

    @tf.function
    def loss_fn(self):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
        loss = (
            tf.reduce_sum(tf.square(self.u - u_pred))
            + tf.reduce_sum(tf.square(self.v - v_pred))
            + tf.reduce_sum(tf.square(f_u_pred))
            + tf.reduce_sum(tf.square(f_v_pred))
        )
        return loss

    def get_weights(self):
        w_list = [w.numpy().flatten() for w in self.weights]
        b_list = [b.numpy().flatten() for b in self.biases]
        l_list = [self.lambda_1.numpy(), self.lambda_2.numpy()]
        return np.concatenate(w_list + b_list + l_list)

    def set_weights(self, params_flat):
        shapes = (
            [w.shape.as_list() for w in self.weights]
            + [b.shape.as_list() for b in self.biases]
            + [[1], [1]]
        )
        sizes = [np.prod(shape) for shape in shapes]
        splits = np.cumsum(sizes[:-1])
        flat_params = np.split(params_flat, splits)

        idx = 0
        for w in self.weights:
            w.assign(tf.reshape(flat_params[idx], w.shape))
            idx += 1
        for b in self.biases:
            b.assign(tf.reshape(flat_params[idx], b.shape))
            idx += 1
        self.lambda_1.assign(tf.reshape(flat_params[idx], [1]))
        idx += 1
        self.lambda_2.assign(tf.reshape(flat_params[idx], [1]))

    def callback(self, params_flat):
        loss = self.loss_fn().numpy()
        lambda_1 = self.lambda_1.numpy()[0]
        lambda_2 = self.lambda_2.numpy()[0]
        print(f"Loss: {loss:.3e}, l1: {lambda_1:.3f}, l2: {lambda_2:.5f}")

    def train_Adam(self, nIter):
        optimizer = tf.keras.optimizers.Adam()

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss = self.loss_fn()
            grads = tape.gradient(
                loss, self.weights + self.biases + [self.lambda_1, self.lambda_2]
            )
            optimizer.apply_gradients(
                zip(grads, self.weights + self.biases + [self.lambda_1, self.lambda_2])
            )
            return loss

        print("Training with Adam...")
        start_time = time.time()
        for it in range(nIter):
            loss = train_step().numpy()
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print(f"It: {it}, Loss: {loss:.3e}, Time: {elapsed:.2f}")
                start_time = time.time()

    def train_LBFGS(self):
        print("Training with L-BFGS-B...")

        def obj_func(params_flat):
            with tf.GradientTape() as tape:
                self.set_weights(params_flat)
                loss = self.loss_fn()
            grads_flat = tape.gradient(loss, self.trainable_variables)
            grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads_flat], axis=0)
            return loss.numpy().astype(np.float64), grads_flat.numpy().astype(
                np.float64
            )

        result = minimize(
            fun=obj_func,
            x0=self.get_weights(),
            method="L-BFGS-B",
            jac=True,
            callback=self.callback,
            options={"maxiter": 50000, "ftol": 1e-8, "gtol": 1e-8},
        )

        self.set_weights(result.x)

    def predict(self, x_star, y_star, t_star):
        x = tf.constant(x_star, dtype=tf.float32)
        y = tf.constant(y_star, dtype=tf.float32)
        t = tf.constant(t_star, dtype=tf.float32)
        u, v, p, _, _ = self.net_NS(x, y, t)
        return u.numpy(), v.numpy(), p.numpy()


# —————————————— Main Training Loop ——————————————

if __name__ == "__main__":
    N_train = 5000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    data = loadmat("../Data/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2

    N, T = X_star.shape[0], t_star.shape[0]
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (N, 1))

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

    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train_Adam(10000)
    model.train_LBFGS()

    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap][:, None]
    v_star = U_star[:, 1, snap][:, None]
    p_star = P_star[:, snap][:, None]

    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

    lambda_1_value = model.lambda_1.numpy()[0]
    lambda_2_value = model.lambda_2.numpy()[0]

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

    # Add plotting code here if needed...
