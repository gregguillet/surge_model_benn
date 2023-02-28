from dataclasses import dataclass
from typing import Tuple, Sequence
from glacier_surge_model import (
    SurgeModel,
    State,
    ProblemUnits,
    EquilibriumPoint,
    ModelJacobian,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)


# ----------------- Flow Plot -----------------
@dataclass(frozen=True)
class ModelBehaviorFromEigVal:
    model: "Model"

    def is_periodic(self):
        eq_point = EquilibriumPoint(self.model).coordinates()
        eigv = ModelJacobian(self.model, eq_point).eigenval()
        # print(eigv)
        if np.all(np.isreal(eigv)):
            # assert np.all(eigv <= 0)
            return False
        ## From now on eigenvalues are complex - We now want to check that real parts are positive
        # assert np.all(np.imag(eigv) != 0.0)
        return True


@dataclass(frozen=True)
class ModelBehaviorFromVFieldIndex:

    model: "Model"

    path_radius: float = 1e-4

    path_resol: int = 36

    def _path_points_around(
        self, point: Tuple[float, float]
    ) -> Sequence[Tuple[float, float]]:

        "Return points along a closed path around given `point`"

        theta = np.linspace(0, 2.0 * np.pi, self.path_resol, endpoint=True)

        xs = point[0] + self.path_radius + np.cos(theta)

        ys = point[1] + self.path_radius + np.sin(theta)

        return zip(xs, ys)

    def _vectors_around(self, point: Tuple[float, float]):

        # Compute vector field at each point, and normalize to unit length

        vec = np.array(
            [self.model.model(x, y) for (x, y) in self._path_points_around(point)]
        )

        norm = np.sqrt(np.sum(vec ** 2, axis=1))

        vec = vec / norm[:, None]

        return vec

    def _index_around(self, point: Tuple[float, float]) -> float:

        """Estimate the index of the flow of the `model` around `point`

        See https://en.wikipedia.org/wiki/Vector_field#Index_of_a_vector_field"""

        # Compute angle between successive unit vectors, by taking:

        # cross(v_{n}, v_{n+1}) = sin(angle(v_{n}, v_{n+1}))

        vec = self._vectors_around(point)

        v_n, v_np1 = vec[:-1, :], vec[1:, :]

        cos_alpha = np.sum(v_n * v_np1, axis=1)

        sin_alpha = np.cross(v_n, v_np1)

        d_alpha = np.arctan2(sin_alpha, cos_alpha)

        # index is alpha/2pi

        return np.sum(d_alpha) / (2.0 * np.pi)

    def _index(self) -> float:

        return self._index_around(EquilibriumPoint(self.model).coordinates())

    def is_periodic(self):

        """Determine if the flow of the given `model` is periodic around `eq_point`

        using the vector field index. See _index(), and also Section 2 of:

        http://www.cds.caltech.edu/archive/help/uploads/wiki/files/224/cds140b-perorb.pdf

        """

        # Flow is periodic if index around equilibrium point is close to +/-1

        # Take absolute value since sign depends on which direction the loop is turning,

        # which is arbitrary

        return np.isclose(abs(self._index()), 1.0, atol=1e-2)


def sum_y_hat(sin_theta, ell_hat):
    model = SurgeModel(
        Ta_hat=-0.8,
        adot_hat=0.7,
        ell_hat=ell_hat,
        sin_theta=sin_theta,
        beta_enabled=True,
    )
    t_hat, y_hat = model.integrate(t_hat_max=20, y0=[0.615, 1.133], n=100000)
    H_hat = y_hat[:, 0]
    E_hat = y_hat[:, 1]

    sum_multiplier = np.count_nonzero(E_hat > np.max(E_hat)-(0.004*np.std(E_hat)))
    return sum_multiplier

def create_geometry_state_space():
    sin_theta_list = np.linspace(0.001, 3.5, 100, endpoint=True) * 0.1
    ell_hat_list = np.linspace(0.001, 2.0, 100, endpoint=True)
    sum_yhat_array = np.array(
        [[sum_y_hat(s, ell) for ell in ell_hat_list] for s in sin_theta_list]
    )
    X, Y = np.meshgrid(ell_hat_list, sin_theta_list)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, sum_yhat_array, 15, cmap="RdBu_r", norm=MidpointNormalize(midpoint=0, vmin=np.min(sum_yhat_array), vmax=np.max(sum_yhat_array)))
    cbar = fig.colorbar(cs)
    plt.clabel(cs, inline=1, fontsize=12)
    plt.show()


modelA = SurgeModel(Ta_hat=-0.8, adot_hat=0.23, ell_hat=1.0653, beta_enabled=True)
modelB = SurgeModel(
    Ta_hat=-0.8, adot_hat=0.4, ell_hat=1.0, sin_theta=0.05, beta_enabled=True
)
modelC = SurgeModel(
    Ta_hat=-0.8, adot_hat=0.7, ell_hat=1.0, sin_theta=0.05, beta_enabled=True
)


modelTests = SurgeModel(
    Ta_hat=-0.8, adot_hat=0.7, ell_hat=1.67, sin_theta=0.066, beta_enabled=True
)

n = 257
H_hat, E_hat = np.linspace(0, 3, n + 1), np.linspace(-1, 2, n)
model = modelTests
#create_geometry_state_space()

### Figures

vectors = np.array([[model.model(ix, iy) for iy in E_hat] for ix in H_hat])
dHdt_hat = vectors[:, :, 0]
dEdt_hat = vectors[:, :, 1]
norm = np.sqrt(dHdt_hat ** 2 + dEdt_hat ** 2)
dHdt_hat = dHdt_hat / norm
dEdt_hat = dEdt_hat / norm

subsampling_factor = 6
# ----------------- QUIVER -----------------
plt.figure()
plt.quiver(
    H_hat[::subsampling_factor],
    E_hat[::subsampling_factor],
    dHdt_hat[::subsampling_factor, ::subsampling_factor].T,
    dEdt_hat[::subsampling_factor, ::subsampling_factor].T,
    pivot="middle",
    color="gray",
)
plt.contour(H_hat, E_hat, dHdt_hat.T, levels=[0], colors="b")
plt.contour(H_hat, E_hat, dEdt_hat.T, levels=[0], colors="k", ls=":")

# plt.axis("equal")
plt.xlim(0, 3)
plt.ylim(-1, 2)
plt.xlabel("H/H0")
plt.ylabel("E/E0")
# -------------------- SOLVE ODE ---------------

t_hat, y_hat = model.integrate(
    t_hat_max=20, y0=[0.6, 1.5], n=100000
)  # [0.615, 1.133][1.5, 0.14][0.615, 1.133]
H_hat = y_hat[:, 0]
E_hat = y_hat[:, 1]


states = [State.from_hat(H_hat_, E_hat_) for H_hat_, E_hat_ in zip(H_hat, E_hat)]
u = [model.u(s) / ProblemUnits.u0 for s in states]
N = [model.N(s) / ProblemUnits.N0 for s in states]
plt.plot(H_hat, E_hat, color="r")
# plt.plot(eq_point[0], eq_point[1], "ro")


#### Plot Fluxes ####
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)

ax1.plot(t_hat, H_hat)
# ax1.set_ylim(0, 2)
ax1.set_ylabel("H/H0")


ax2.plot(t_hat, E_hat)
# ax2.set_ylim(-1, 2)
ax2.set_ylabel("E/E0")

ax3.plot(t_hat, u, color="orange")
ax3.semilogy()
# ax3.set_ylim(1e-2, 1e2)
ax3.set_ylabel("u/u0")

ax4.plot(t_hat, (H_hat * ProblemUnits.H0 / N) ** 3)
ax4.set_ylabel("N/N0")
# plt.xlim(2, 20)


plt.show()
