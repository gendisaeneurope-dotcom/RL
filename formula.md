\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\title{Reward Function Formulations for AP Postural Control Candidates}
\author{}
\date{}
\begin{document}
\maketitle

\section{Base Formulation (Arditi et al., 2024)}

The original Ecological Cost (EC) function trades off effort against safety via a convex combination controlled by $\omega \in [0,1]$:

\begin{equation}
EC_\omega(s,u) = \omega \cdot EF_u + (1-\omega) \cdot COM_x
\end{equation}

where effort is defined as
\begin{equation}
EF_u = K\|u\|^2 = K(u_1^2 + u_2^2 + u_3^2)
\end{equation}

with $K$ a scaling constant, and $COM_x$ is the horizontal center-of-mass offset used as the safety proxy (leaning forward increases distance from the base-of-support edge). The overall reward combines this running cost with terminal success/failure conditions based on body height $h \in [0,1]$:

\begin{equation}
R(s,u) =
\begin{cases}
\text{success bonus} & \text{if } h \geq 0.99 \text{ and } |\dot{h}| \leq 0.1 \\
\text{failure penalty} & \text{if joint limit violated, fallen, or timeout} \\
h - EC_\omega(s,u) & \text{otherwise}
\end{cases}
\end{equation}

Higher $\omega$ (near 1) promotes effort minimization; lower $\omega$ (near 0) promotes safety-seeking (larger COM offset).

\section{Adapted Formulation for AP Target-Tracking Task}

The candidates replace the stand-up task with an anterior-posterior (AP) CoM target-tracking task. The safety proxy $COM_x$ is replaced by a \emph{tracking} term toward a target $x^*$, and an optional \emph{safety} stability term is added on top.

\subsection{Common Terms}

Effort (energy) term, analogous to $EF_u$:
\begin{equation}
\text{energy} = -\omega \cdot \frac{1}{N}\sum_{i=1}^{N} u_i^2
\end{equation}

Normalized tracking error and height-like shaping variable:
\begin{equation}
d_{\text{norm}} = \frac{|com_x - x^*|}{S}, \qquad
\hat{d} = \min(d_{\text{norm}}, d_{\text{cap}}), \qquad
h = 1 - \hat{d}
\end{equation}
where $S$ is the target-span normalization constant and $d_{\text{cap}}$ caps the normalized distance.

Tracking reward (piecewise, mirroring Arditi's terminal/running split):
\begin{equation}
\text{tracking} =
\begin{cases}
R_{\text{fail,base}} + R_{\text{fail,slope}}\,(1-h) & \text{failed} \\
R_{\text{success}} & \text{success: } |com_x - x^*| < \epsilon_{pos},\ |\dot{com}_x| < \epsilon_{vel} \\
(1-\omega)\,h & \text{otherwise}
\end{cases}
\end{equation}

Optional shaping bonus (potential-based, disabled in current runs, \texttt{USE\_SHAPING=False}):
\begin{equation}
\text{shaping} = \lambda_{\text{shape}} \left( \hat{d}_{t-1} - \hat{d}_t \right)
\end{equation}

\subsection{Candidate 1 (Baseline "F"): Energy + Tracking Only}
\begin{equation}
R_1 = \text{energy} + \text{tracking}
\end{equation}
No stability/safety term. This is the direct analogue of Arditi's $EC_\omega$ but with $COM_x$ safety replaced entirely by target-tracking.

\subsection{Candidate 2: Energy + Tracking + XCoM Safety}

The Extrapolated Center of Mass (XCoM) introduces the inverted-pendulum natural frequency $\omega_0 = \sqrt{g / z_{com}}$, where $z_{com}$ is CoM height:
\begin{equation}
xcom_x = com_x + \frac{\dot{com}_x}{\omega_0}
\end{equation}

The safety term penalizes CoM \emph{velocity-driven instability} continuously (non-zero even at rest):
\begin{equation}
\text{instability} = \frac{\left| \dot{com}_x / \omega_0 \right|}{L_{\text{base}}}, \qquad
\text{safety}_{\text{XCoM}} = -w_{\text{safety}} \cdot \text{instability}
\end{equation}
where $L_{\text{base}}$ is the half-length of the base of support (foot). Episode termination additionally triggers if $|xcom_x| > L_{\text{base}}$.

\begin{equation}
R_2 = \text{energy} + \text{tracking} + \text{safety}_{\text{XCoM}}
\end{equation}

\subsection{Candidate 3: Energy + Tracking + Capture-Point Safety}

Same extrapolated quantity, termed the capture point:
\begin{equation}
cp = com_x + \frac{\dot{com}_x}{\omega_0}
\end{equation}

Unlike XCoM's term, the capture-point safety penalizes only the \emph{excess} beyond the base-of-support boundary (zero gradient when safely inside):
\begin{equation}
\text{excess} = \frac{\max(0,\ |cp| - L_{\text{base}})}{L_{\text{base}}}, \qquad
\text{safety}_{\text{CP}} = -w_{\text{safety}} \cdot \text{excess}^2
\end{equation}

\begin{equation}
R_3 = \text{energy} + \text{tracking} + \text{safety}_{\text{CP}}
\end{equation}

\section{Summary Table}

\begin{table}[h]
\centering
\begin{tabular}{@{}lll@{}}
\toprule
Candidate & Reward Terms & Characteristics \\
\midrule
1 (F, baseline) & energy + tracking & none \\
2 (XCoM) & energy + tracking + safety$_{\text{XCoM}}$ & continuous, active at all times \\
3 (Capture-point) & energy + tracking + safety$_{\text{CP}}$ & sparse, active only near boundary \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
