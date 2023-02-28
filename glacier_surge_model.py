import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.misc import derivative


def inverse_machine_epsilon(y):
    # change x/y to x*y/(float_epsilon**2+y**2)
    float_epsilon = 0  # 1e-12
    return y / (float_epsilon**2 + y**2)


def div(x, y):
    return x * inverse_machine_epsilon(y)


SECONDS_PER_YEAR = 3600 * 24 * 365.15


class Cst:
    # Glenn's flow Law
    A = 2.4e-25  # Creep parameter
    n = 3  # GF L exponent

    # Standard physical constants
    g = 10.0  # *(u.m)*(u.second**-2) - Gravity acceleration
    rho = 916  # *(u.kilogram*(u.m**-3)) - Ice density

    # Thermodynamics
    cp = 2000  # **(u.Joule)*(u.kilogram**-1)*(u.Kelvin**-1) - Specific heat capicity of ice
    L = 3.3e5  # *(u.Joule)*(u.kilogram**-1) - Latent heat of fusion
    k = 2.1  # *(u.Watt)*(u.m**-1)*'(u.Kelvin**-1)' - Thermal conductivity
    DDF = 0.1 / SECONDS_PER_YEAR  # *(u.m)*(u.yr**-1)*'(u.Kelvin**-1) Degree day factor
    G = 0.06  # w m-2

    K = 2.3e-47  ## 2.3e-47#0.125e-46  # *((3600*24*365.15)**9)#*((u.kilogram**-5)*(u.m**2)*(u.second**9))

    C = 9.2e13  # *((u.Pascal)*(u.Joule)*(u.m**-2))

    # Weertman friction law
    q = 1.0
    p = 1 / 3
    R = 15.7  # *((u.m**(-1/3))*(u.second**(1/3))) - Roughness Coeff

    # Problem-specific csts
    d = 10  # *u.m thickness of the basal layer
    Tm = 0  # *u.Celsius - Melting temperature
    T_offset = -10  # *u.Celsius - Temperature below which there is no melting
    w0 = 0.6  # Average depth of water stored at the bed

    Wc = 1000.0  # *u.m Space width between conduits

    # Velocities
    u1 = 0 / SECONDS_PER_YEAR  # m/a
    u2 = 5000 / SECONDS_PER_YEAR  # m/a

    # Dimensionless parameters
    alpha = 5.0


def K_tilde(sin_theta):
    return (
        Cst.K
        * (Cst.rho ** (Cst.alpha - 1))
        * (Cst.L**Cst.alpha)
        / (Cst.g * sin_theta)
    )


class ZoetIversonCst:  # Zoet and Iverson friction law
    R = 15  # mm
    till_angle = 32  #
    nu = 32.0e12  # Pa*s
    Nf = 33
    k = 0.1
    a = 0.25
    till_exponent = 5  ## Till exponent
    K = 2.55  # W m-1 K-1
    L = 3e8  # J*m3 Volumetric latent heat of ice
    Cp = 7.4e-8  # K Pa-1 depression of the metling temperature of ice with pressure
    C1 = Cp * K / L  ## Regelation parameter


class ProblemUnits:
    T0 = 10  # E0/(Cst.rho*Cst.cp*Cst.d)#*u.Kelvin
    a0 = 1.0 / SECONDS_PER_YEAR  # * m/s (u.meter*((u.second)**-1))
    ell0 = 10000  # *(u.meter)
    t0 = 200 * SECONDS_PER_YEAR
    sin_theta = 0.05  # Sine of the slope angle

    E0 = 1.8e8  # E0= 1.8e8 (Cst.g*Cst.sin_theta*a0*(l0**2))/(Cst.L*Cst.K))**(1/Cst.alpha)#*(u.Joule/u.m**2)
    H0 = 200  # ((Cst.R*(Cst.C**Cst.q)*(a0**Cst.p)*(l0**Cst.p))/(Cst.rho*Cst.g*Cst.sin_theta*(E0**Cst.q)))**(1/(Cst.p+1))
    u0 = 50 / SECONDS_PER_YEAR
    N0 = 0.5e6  ## Pascals


@dataclass(frozen=True)
class State:
    H: float
    E: float

    @classmethod
    def from_hat(cls, H_hat, E_hat):
        return cls(H=H_hat * ProblemUnits.H0, E=E_hat * ProblemUnits.E0)


class ModelExpression(ABC):
    @abstractmethod
    def __call__(self, state: State) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstExpr(ModelExpression):
    value: float

    def __call__(self, state: State) -> float:
        return self.value


@dataclass(frozen=True)
class HEvolEquation(ModelExpression):
    """Equation 1"""

    a_dot: float
    m_dot: float
    Qi: ModelExpression
    ell: float

    def __call__(self, state: State) -> float:
        return (self.a_dot - self.m_dot) - (self.Qi(state) / self.ell)


@dataclass(frozen=True)
class IceFlux(ModelExpression):
    """Qi, Equation 6"""

    u: ModelExpression
    A: float
    rho_g_sin_theta: float
    n: float

    def __call__(self, state: State) -> float:
        glen_law = 2 * self.A * (self.rho_g_sin_theta**self.n) / (self.n + 2)
        return state.H * self.u(state) + glen_law * state.H ** (self.n + 2)


@dataclass(frozen=True)
class IceVelocity(ModelExpression):
    """u, Equation 8"""

    rho_g_sin_theta: float
    R: float
    N: ModelExpression
    p: float
    q: float

    def __call__(self, state: State) -> float:
        inv_p = 1.0 / self.p
        N = self.N(state)
        H = state.H
        u = (
            (self.rho_g_sin_theta / self.R) ** inv_p
            * (H**inv_p)
            * (N ** (-self.q * inv_p))
        )
        return u


@dataclass(frozen=True)
class EffectivePressure(ModelExpression):
    """N, Equation 11"""

    rho: float
    g: float
    C: float

    def __call__(self, state: State) -> float:
        E_plus = plus(state.E)
        if E_plus > 0:
            return min(self.rho * self.g * state.H, self.C / E_plus)
        else:
            return self.rho * self.g * state.H


@dataclass(frozen=True)
class EEvolEquation(ModelExpression):
    tau: ModelExpression
    beta: ModelExpression
    u: ModelExpression
    G: float
    qi: ModelExpression
    rho: float
    Ell: float
    Qw: ModelExpression
    ell: float
    m_dot: float

    def __call__(self, state: State) -> float:
        """Equation 2"""
        u = self.u(state)
        tau_u = self.tau(state) * u
        discharge = (self.rho * self.Ell * self.Qw(state)) / self.ell
        prop_surface_melt = self.rho * self.Ell * self.beta(state) * self.m_dot
        return tau_u + self.G - self.qi(state) - discharge + prop_surface_melt


@dataclass(frozen=True)
class WeertmanStrain(ModelExpression):
    u: ModelExpression
    R: float
    N: ModelExpression
    p: float
    q: float

    def __call__(self, state: State) -> float:
        return self.R * (self.u(state) ** self.p) * (self.N(state) ** self.q)


@dataclass(frozen=True)
class ZoetIversonTransitionVelocity(ModelExpression):
    N: ModelExpression
    R: float
    nu: float
    a: float
    C1: float
    Nf: float
    k: float

    def __call__(self, state: State) -> float:
        k0 = (2 * np.pi) / (4 * self.R)
        numerator_left = 1 / (self.nu * (self.R * self.a) ** 2 * k0**3) + (
            4 * self.C1 / ((self.R * self.a) ** 2 * k0)
        )
        numerator_right = self.Nf * self.N(state)
        denominator = 2 + self.Nf * self.k
        return (numerator_left * numerator_right) / denominator


@dataclass(frozen=True)
class ZoetIversonStrain(ModelExpression):
    ut: ModelExpression
    u: ModelExpression
    N: ModelExpression
    till_angle: float
    slip_exponent: float

    def __call__(self, state: State) -> float:
        return (
            self.N(state)
            * np.tan(self.till_angle)
            * (self.u(state) / (self.u(state) + self.ut(state))) ** (1 / self.slip_exponent)
        )


@dataclass(frozen=True)
class EffectiveCooling(ModelExpression):
    """Equation 9"""

    k: float
    T: ModelExpression
    Tm: float
    Ta: float

    def __call__(self, state: State) -> float:
        return (
            self.k
            * (minus(self.T(state) - self.Tm) - minus(self.Ta - self.Tm))
            / state.H
        )


@dataclass(frozen=True)
class BasalTemperature(ModelExpression):
    """Equation 5"""

    rho: float
    cp: float
    d: float

    def __call__(self, state: State) -> float:
        return minus(state.E) / (self.rho * self.cp * self.d)


@dataclass(frozen=True)
class SingleComponentDrainage(ModelExpression):
    """Qw, Equation 12"""

    K_tilde: float
    alpha: float
    sin_theta: float

    def __call__(self, state: State) -> float:
        K = (self.K_tilde * Cst.g * self.sin_theta) / (
            (Cst.rho ** (self.alpha - 1)) * (Cst.L**self.alpha)
        )
        return K * (plus(state.E) ** self.alpha)


@dataclass(frozen=True)
class TwoFactorComponentDrainage(ModelExpression):
    """Qw"""

    Qw1: SingleComponentDrainage
    Qw2: SingleComponentDrainage

    def __call__(self, state: State) -> float:
        # assert self.Qw1.alpha < Cst.q / Cst.p
        # assert self.Qw2.alpha > Cst.q / Cst.p
        return self.Qw1(state) + self.Qw2(state)


@dataclass(frozen=True)
class DoubleComponentDrainage(ModelExpression):
    """Equation 13"""

    K: float
    alpha: float
    phi: ModelExpression
    Kc: float
    Wc: float
    rho_g_sin_theta: float
    S: ModelExpression

    def __call__(self, state: State) -> float:
        discharge = self.K * (plus(state.E) ** self.alpha)
        conduit = (
            self.phi(state)
            * (self.Kc / self.Wc)
            * (self.rho_g_sin_theta**0.5)
            * (self.S(state) ** (4 / 3))
        )
        return discharge + conduit


@dataclass(frozen=True)
class Beta(ModelExpression):
    """Equation 16"""

    u: ModelExpression
    u1: float
    u2: float

    def __call__(self, state: State) -> float:
        return min(max(0.0, (self.u(state) - self.u1) / (self.u2 - self.u1)), 1.0)


@dataclass
class SurgeModel:
    Ta_hat: float
    adot_hat: float
    ell_hat: float
    sin_theta: float = 0.1
    beta_enabled: bool = True

    def model(self, H_hat, E_hat):
        H = H_hat * ProblemUnits.H0
        E = E_hat * ProblemUnits.E0

        state = State(H=H, E=E)
        dHdt = self.H_dot(state)
        dEdt = self.E_dot(state)
        return np.array([dHdt / ProblemUnits.H0, dEdt / ProblemUnits.E0])

    def __post_init__(self):
        Ta = self.Ta_hat * ProblemUnits.T0
        self.N = EffectivePressure(Cst.rho, Cst.g, Cst.C)
        mdot = Cst.DDF * plus(Ta - Cst.T_offset)  ## Eq.3
        adot = self.adot_hat * ProblemUnits.a0

        self.u = IceVelocity(
            rho_g_sin_theta=Cst.rho * Cst.g * self.sin_theta,
            R=Cst.R,
            N=self.N,
            p=Cst.p,
            q=Cst.q,
        )

        Qw = TwoFactorComponentDrainage(
            Qw1=SingleComponentDrainage(
                K_tilde=K_tilde(self.sin_theta), sin_theta=self.sin_theta, alpha=2
            ),  # alpha < q/p
            Qw2=SingleComponentDrainage(
                K_tilde=K_tilde(self.sin_theta), sin_theta=self.sin_theta, alpha=4
            ),  # alpha > q/p
        )
        T = BasalTemperature(rho=Cst.rho, cp=Cst.cp, d=Cst.d)
        qi = EffectiveCooling(k=Cst.k, T=T, Tm=Cst.Tm, Ta=Ta)

        self.ut = ZoetIversonTransitionVelocity(
            N=self.N,
            R=ZoetIversonCst.R,
            nu=ZoetIversonCst.nu,
            a=ZoetIversonCst.a,
            C1=ZoetIversonCst.C1,
            Nf=ZoetIversonCst.Nf,
            k=ZoetIversonCst.k,
        )

        tau = ZoetIversonStrain(
            ut=self.ut,
            u=self.u,
            N=self.N,
            till_angle=ZoetIversonCst.till_angle,
            slip_exponent=ZoetIversonCst.till_exponent,
        )
        # tau = WeertmanStrain(u=self.u, R=Cst.R, N=self.N, p=Cst.p, q=Cst.q)

        self.H_dot = HEvolEquation(
            a_dot=adot,
            m_dot=mdot,
            Qi=IceFlux(
                A=Cst.A,
                n=Cst.n,
                rho_g_sin_theta=Cst.rho * Cst.g * self.sin_theta,
                u=self.u,
            ),
            ell=self.ell_hat * ProblemUnits.ell0,
        )
        if self.beta_enabled:
            beta = Beta(self.u, u1=Cst.u1, u2=Cst.u2)
        else:
            beta = ConstExpr(0.0)
        self.E_dot = EEvolEquation(
            tau=tau,
            u=self.u,
            G=Cst.G,
            qi=qi,
            rho=Cst.rho,
            Ell=Cst.L,
            Qw=Qw,
            ell=self.ell_hat * ProblemUnits.ell0,
            m_dot=mdot,
            beta=beta,
        )

    def integrate(self, t_hat_max, y0=[1.5, 0.0], n=1000):
        t_hat = np.linspace(0, t_hat_max, n, endpoint=True)

        def func(y, t):
            H_hat, E_hat = y
            return self.model(H_hat, E_hat) * ProblemUnits.t0

        y = odeint(func=func, y0=y0, t=t_hat)
        return t_hat, y


@dataclass(frozen=True)
class EquilibriumPoint:
    model: "Model"
    H_hat0: float = 1
    E_hat0: float = 1

    def coordinates(self):
        f = lambda point: self.model.model(*point)
        result = fsolve(f, (self.H_hat0, self.E_hat0))
        return result


@dataclass(frozen=True)
class ModelJacobian:
    model: "Model"
    point: "point"

    def matrix(self):
        H0, E0 = self.point
        f_hh = lambda h: self.model.model(h, E0)[0]
        f_he = lambda h: self.model.model(h, E0)[1]

        f_eh = lambda e: self.model.model(H0, e)[0]
        f_ee = lambda e: self.model.model(H0, e)[1]

        de = lambda f: derivative(f, E0, n=1, dx=1e-4)
        dh = lambda f: derivative(f, H0, n=1, dx=1e-4)

        return np.array([[dh(f_hh), dh(f_he)], [de(f_eh), de(f_ee)]])

    def eigenval(self):
        m = self.matrix()
        return np.linalg.eigvals(m)


def plus(x):
    return max(x, 0)


def minus(x):
    return min(x, 0)
