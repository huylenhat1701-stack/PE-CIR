# -*- coding: utf-8 -*-
"""Script to insert Section 3 (Method) into paper/experiments.tex with full academic rigor."""

section3_text = r"""
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % OVERLEAF NAVIGATION NOTE:
% % ---------------------------------------------------------------
% % This paper is structured into modular sections for Overleaf:
% %   - Abstract (Lines ~80)
% %   - Section 1: Introduction (Lines ~125)
% %   - Section 3: Method (Below)
% %       * 3.1 Problem Formulation
% %       * 3.2 Preserve--Edit Factorization
% %       * 3.3 Counterfactual Constraint Construction
% %       * 3.4 Counterfactual Constraint Learning
% %       * 3.5 Candidate-Level Constraint Verification
% %       * 3.6 Verifier-Based Candidate Reranking
% %   - Section 4: Experiments (See SECTION 4 banner)
% %   - Section 5: Conclusion (See SECTION 5 banner)
% % NOTE: Section 2 (Related Work) is omitted in this manuscript draft.
% % We set \setcounter{section}{2} below so that the numbering
% % automatically maps to Section 3 (Method) and Section 4 (Experiments).
% % When adding Section 2, simply delete \setcounter{section}{2}.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setcounter{section}{2}

% =================================================================
% =================================================================
% SECTION 3: METHOD
% =================================================================
% =================================================================
\section{Method}
\label{sec:method}

In this section, we present the CC-CIR framework (\textbf{C}ounterfactual
\textbf{C}onstraint learning for \textbf{C}omposed \textbf{I}mage
\textbf{R}etrieval).
We begin with the mathematical problem formulation and constraint decomposition
in Section~\ref{sec:method-formulation}.
Section~\ref{sec:method-factorization} introduces the preserve--edit
factorization mechanism and Stage-1 composed query construction over a frozen
vision--language backbone.
In Section~\ref{sec:method-cf-construction}, we formalise the generation of
structured counterfactual hard negatives (CF-A and CF-B).
Section~\ref{sec:method-cf-learning} defines the counterfactual margin ranking
objective and composite training formulation.
Finally, Sections~\ref{sec:method-verification} and~\ref{sec:method-reranking}
detail the architecture, objective, and inference-time reranking protocol of the
candidate-level constraint verifier.

% -----------------------------------------------------------------
% 3.1. Problem Formulation
% -----------------------------------------------------------------
\subsection{Problem Formulation}
\label{sec:method-formulation}

\subsubsection{Composed Image Retrieval}
\label{sec:method-cir}
Let $\mathcal{I}$ denote the open-domain image space and $\mathcal{T}$ denote the
space of natural-language text descriptions.
In Composed Image Retrieval~(CIR), a user provides a multimodal query
$q = (I_r, T_m)$, where $I_r \in \mathcal{I}$ represents a visual reference
image and $T_m \in \mathcal{T}$ specifies a relative modification instruction.
Given an unannotated search gallery $\mathcal{G} = \{I_c\}_{c=1}^{|\mathcal{G}|}
\subset \mathcal{I}$, the objective is to retrieve the target image
$I_t^* \in \mathcal{G}$ that simultaneously retains the visual context of $I_r$
and reflects the modifications requested in $T_m$.
Performance is conventionally measured by Recall at rank $k$ (R@$k$) over the
entire gallery $\mathcal{G}$, as well as Group Recall at rank $k$ (GR@$k$) within
local subsets of visually similar distractors~\citep{liu2021cirr}.

\subsubsection{Preservation and Modification Constraints}
\label{sec:method-constraints}
A composed query implicitly enforces two distinct semantic constraints on any
valid target candidate:
\begin{enumerate}
    \item \textbf{Preservation constraint ($\mathcal{C}_P$):} Attributes, visual
    identity, and contextual features of $I_r$ that are not explicitly targeted
    for alteration must remain invariant in the target image.
    Formally, let $\mathcal{A}(I)$ represent the semantic attribute set of image
    $I$, and let $\mathcal{A}_{\mathrm{mod}} \subset \mathcal{A}(I_r)$ denote the
    attributes designated for modification by $T_m$.
    Then $\mathcal{C}_P$ requires that:
    \begin{equation}
        \mathcal{A}(I_t^*) \cap \left(\mathcal{A}(I_r) \setminus \mathcal{A}_{\mathrm{mod}}\right)
        \;=\; \mathcal{A}(I_r) \setminus \mathcal{A}_{\mathrm{mod}}.
        \label{eq:preservation-constraint}
    \end{equation}

    \item \textbf{Modification constraint ($\mathcal{C}_E$):} The semantic
    alterations dictated by $T_m$ must be faithfully incorporated into the
    target image.
    Letting $\mathcal{A}'_{\mathrm{mod}}$ denote the modified attribute values,
    $\mathcal{C}_E$ requires:
    \begin{equation}
        \mathcal{A}'_{\mathrm{mod}} \;\subseteq\; \mathcal{A}(I_t^*).
        \label{eq:edit-constraint}
    \end{equation}
\end{enumerate}
Standard composed retrieval models fuse $I_r$ and $T_m$ into a monolithic query
vector $q$ via unconstrained cross-attention or token injection, ranking candidates
purely by cosine similarity without explicitly decomposing the target semantics into
$\mathcal{C}_P$ and $\mathcal{C}_E$.
Under unconstrained optimization, models frequently suffer from two failure modes:
\emph{reference bias} (retrieving candidates that satisfy $\mathcal{C}_P$ but ignore
$\mathcal{C}_E$) or \emph{edit bias} (retrieving candidates that satisfy
$\mathcal{C}_E$ but violate $\mathcal{C}_P$ by depicting an entirely different
object or scene).

% -----------------------------------------------------------------
% 3.2. Preserve--Edit Factorization
% -----------------------------------------------------------------
\subsection{Preserve--Edit Factorization}
\label{sec:method-factorization}

\subsubsection{Frozen Vision--Language Backbone}
\label{sec:method-backbone}
To establish robust multimodal representations while preventing catastrophic
forgetting, we build our system upon a pre-trained CLIP
ViT-L-14-quickgelu~\citep{radford2021clip} backbone.
Let $E_{\mathrm{img}}: \mathcal{I} \to \mathbb{R}^d$ and
$E_{\mathrm{txt}}: \mathcal{T} \to \mathbb{R}^d$ denote the pre-trained visual
and textual encoders, respectively, where $d = 768$.
All backbone parameters are strictly frozen throughout training.
For any query pair $(I_r, T_m)$, we extract $\ell_2$-normalized global embeddings:
\begin{equation}
    z_r = \frac{E_{\mathrm{img}}(I_r)}{\|E_{\mathrm{img}}(I_r)\|_2} \in \mathbb{R}^d,
    \qquad
    t_m = \frac{E_{\mathrm{txt}}(T_m)}{\|E_{\mathrm{txt}}(T_m)\|_2} \in \mathbb{R}^d.
    \label{eq:backbone-embeddings}
\end{equation}
Similarly, each gallery image $I_c \in \mathcal{G}$ is embedded offline as
$z_c = E_{\mathrm{img}}(I_c) / \|E_{\mathrm{img}}(I_c)\|_2 \in \mathbb{R}^d$.

\subsubsection{Preserve--Edit Factorization}
\label{sec:method-factorizer}
The \texttt{PreserveEditFactorizer} decomposes the global reference embedding
$z_r$ into two disentangled components conditioned on the text instruction $t_m$.
First, $z_r$ is expanded into $M = 8$ learnable semantic slots
$S \in \mathbb{R}^{M \times d}$:
\begin{equation}
    S^{(0)} = \mathrm{reshape}\left(W_{\mathrm{slot}} z_r,\, (M, d)\right),
    \qquad
    S = \mathrm{LayerNorm}\left(S^{(0)} + \mathbf{1}_M z_r^\top\right),
    \label{eq:slot-expansion}
\end{equation}
where $W_{\mathrm{slot}} \in \mathbb{R}^{(M \cdot d) \times d}$ is a trainable
linear projection and $\mathbf{1}_M \in \mathbb{R}^{M \times 1}$ broadcasts the
reference embedding across all slots.

Second, a text-conditioned gating module determines the relevance of each slot
to the requested edit.
For each slot index $i \in \{1, \dots, M\}$, intermediate gate representations
$H_i \in \mathbb{R}^{d_h}$ and scalar gating weights $g_i \in (0, 1)$ are computed via:
\begin{equation}
    H_i = \tanh\left(W_z S_i + W_t t_m\right),
    \qquad
    g_i = \sigma\left(w^\top H_i\right),
    \label{eq:gating}
\end{equation}
where $W_z \in \mathbb{R}^{d_h \times d}$, $W_t \in \mathbb{R}^{d_h \times d}$,
$w \in \mathbb{R}^{d_h}$, $d_h = 512$ is the hidden layer dimension, and
$\sigma(\cdot)$ denotes the logistic sigmoid.
The gate activation $g_i$ quantifies the alignment of slot $S_i$ with the edit
instruction $T_m$.
Complementary pooling across the slots yields the factorized edit representation
$e \in \mathbb{R}^d$ and preserve representation $p \in \mathbb{R}^d$:
\begin{equation}
    e = \frac{\sum_{i=1}^M g_i S_i}{\sum_{i=1}^M g_i + \epsilon},
    \qquad
    p = \frac{\sum_{i=1}^M (1 - g_i) S_i}{\sum_{i=1}^M (1 - g_i) + \epsilon},
    \label{eq:factorized-pooling}
\end{equation}
where $\epsilon = 10^{-7}$ prevents division by zero.

To prevent representation collapse and promote semantic orthogonality, we train
the factorizer with an auxiliary regularization loss $\mathcal{L}_{\mathrm{fac}}$:
\begin{equation}
    \mathcal{L}_{\mathrm{fac}} = \mathcal{L}_{\mathrm{leak}} + \mathcal{L}_{\mathrm{align}} + 0.1 \cdot \mathcal{L}_{\mathrm{balance}},
    \label{eq:loss-fac}
\end{equation}
where:
\begin{align}
    \mathcal{L}_{\mathrm{leak}} &= \left(\frac{p^\top t_m}{\|p\|_2 \|t_m\|_2}\right)^2,
    \label{eq:loss-leak} \\
    \mathcal{L}_{\mathrm{align}} &= 1 - \frac{\tilde{e}^\top t_m}{\|\tilde{e}\|_2 \|t_m\|_2},
    \label{eq:loss-align} \\
    \mathcal{L}_{\mathrm{balance}} &= \left(\frac{1}{M}\sum_{i=1}^M g_i - 0.5\right)^2.
    \label{eq:loss-balance}
\end{align}
Here, $\mathcal{L}_{\mathrm{leak}}$ suppresses modification information inside the
preserve representation $p$; $\mathcal{L}_{\mathrm{align}}$ aligns the composed
edit $\tilde{e}$ with $t_m$; and $\mathcal{L}_{\mathrm{balance}}$ discourages
trivial gate saturation where all gates collapse towards $0$ or $1$.

\subsubsection{Stage-1 Retrieval Representation}
\label{sec:method-stage1}
The \texttt{EditComposer} module merges the isolated edit slot $e$ with the raw
instruction $t_m$ to construct the composed edit vector $\tilde{e} \in \mathbb{R}^d$:
\begin{equation}
    \tilde{e} = \mathrm{MLP}_{\mathrm{edit}}\left([e;\, t_m]\right),
    \label{eq:composed-edit}
\end{equation}
where $[\cdot; \cdot]$ denotes tensor concatenation along the feature dimension,
and $\mathrm{MLP}_{\mathrm{edit}}$ comprises a $\text{Linear}(2d \to d_h)$ layer,
$\text{GELU}$ activation, and a $\text{Linear}(d_h \to d)$ projection.

The final Stage-1 retrieval query vector $q \in \mathbb{R}^d$ is generated by
linearly projecting and aggregating the preserve, composed-edit, and text features:
\begin{equation}
    q = \mathrm{normalize}\left(W_p p + W_e \tilde{e} + W_m t_m\right),
    \label{eq:stage1-query}
\end{equation}
where $W_p, W_e, W_m \in \mathbb{R}^{d \times d}$ are learnable weight matrices,
and $\mathrm{normalize}(x) = x / \|x\|_2$.
The Stage-1 cosine similarity score between query $q$ and any gallery candidate
$I_c$ with embedding $z_c$ is computed as:
\begin{equation}
    S_G(c) = q^\top z_c.
    \label{eq:stage1-score}
\end{equation}

% -----------------------------------------------------------------
% 3.3. Counterfactual Constraint Construction
% -----------------------------------------------------------------
\subsection{Counterfactual Constraint Construction}
\label{sec:method-cf-construction}

\subsubsection{Motivation}
\label{sec:method-cf-motivation}
In standard contrastive learning pipelines, negative samples are drawn randomly
from within the mini-batch.
Random negatives almost invariably differ from the reference image in both entity
class and background scene, violating both $\mathcal{C}_P$ and $\mathcal{C}_E$
simultaneously (i.e., $P^- E^-$ negatives).
Consequently, a model can successfully discriminate random negatives using trivial
global semantic heuristics, without learning to disentangle the distinct constraint
boundaries.
To force the retrieval model to actively verify both constraints, we synthesize
two structured counterfactual hard negatives for every query tuple:
CF-A ($P^+ E^-$) and CF-B ($P^- E^+$).

\subsubsection{CF-A: Edit-Violation Counterfactual}
\label{sec:method-cf-a}
The CF-A candidate satisfies the preservation constraint $\mathcal{C}_P$ but
violates the edit constraint $\mathcal{C}_E$.
Specifically, CF-A depicts the exact same object entity or category as $I_r$ and
$I_t^+$, but retains the original, pre-modification attribute value $a_{\mathrm{before}}$
instead of incorporating the requested edit $a_{\mathrm{after}}$.
Formally, let $\mathcal{G}_{\mathrm{obj}(I_r)}$ be the subset of gallery images
sharing the object class of $I_r$.
CF-A is selected such that:
\begin{equation}
    I_{\mathrm{CF-A}} \in \mathcal{G}_{\mathrm{obj}(I_r)}, \quad
    a_{\mathrm{before}} \in \mathcal{A}(I_{\mathrm{CF-A}}), \quad
    I_{\mathrm{CF-A}} \neq I_t^+.
    \label{eq:cf-a-def}
\end{equation}
CF-A penalizes models that suffer from reference bias and ignore the edit text.

\subsubsection{CF-B: Preservation-Violation Counterfactual}
\label{sec:method-cf-b}
Conversely, the CF-B candidate satisfies the edit constraint $\mathcal{C}_E$ but
violates the preservation constraint $\mathcal{C}_P$.
CF-B incorporates the modified attribute $a_{\mathrm{after}}$, but manifests this
change on an entirely different object category or visual scene.
Formally, letting $\mathcal{G}_{a_{\mathrm{after}}}$ denote candidates exhibiting
attribute $a_{\mathrm{after}}$, CF-B is selected such that:
\begin{equation}
    I_{\mathrm{CF-B}} \in \mathcal{G}_{a_{\mathrm{after}}}, \quad
    \mathrm{obj}(I_{\mathrm{CF-B}}) \neq \mathrm{obj}(I_r).
    \label{eq:cf-b-def}
\end{equation}
CF-B penalizes models that suffer from edit bias and discard reference identity.

\subsubsection{Counterfactual Pseudo-Labels}
\label{sec:method-cf-labels}
The training dataset is mined from Conceptual Captions 3M
(CC3M)~\citep{changpinyo2021cc12m}.
A rule-based parsing engine matches captions against 37 visual object classes
(e.g., dress, shirt, car, dog, chair) and four attribute categories: colour
(12 values), pattern (7 values), material (9 values), and size/shape (9 values).
From approximately 100{,}000 candidate records scanned, 50{,}000 valid tuples
were mined, yielding a finalized training manifest of 45{,}000 tuples across
13{,}582 unique image paths.
Each training instance forms a five-tuple:
\begin{equation}
    \tau = \left(I_r,\, I_t^+,\, I_{\mathrm{CF-A}},\, I_{\mathrm{CF-B}},\, T_m\right).
    \label{eq:tuple-def}
\end{equation}
For candidate verification, we assign binary truth targets
$\mathbf{y} = (y_P, y_E, y_V)^\top \in \{0, 1\}^3$ reflecting preservation
satisfaction ($y_P$), edit satisfaction ($y_E$), and constraint violation ($y_V$):
\begin{equation}
    \mathbf{y}(I_t^+) = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix},
    \qquad
    \mathbf{y}(I_{\mathrm{CF-A}}) = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix},
    \qquad
    \mathbf{y}(I_{\mathrm{CF-B}}) = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}.
    \label{eq:cf-labels}
\end{equation}

% -----------------------------------------------------------------
% 3.4. Counterfactual Constraint Learning
% -----------------------------------------------------------------
\subsection{Counterfactual Constraint Learning}
\label{sec:method-cf-learning}

\subsubsection{Positive--Counterfactual Ranking}
\label{sec:method-pos-cf-ranking}
In Stage-1 embedding space, the composed query representation $q$ must rank the
positive target $I_t^+$ strictly above both CF-A and CF-B.
To enforce a clear geometric separation, we require that the similarity of the
positive target exceeds that of each counterfactual by at least a predefined
margin $m > 0$:
\begin{equation}
    q^\top z_t^+ - q^\top z_{\mathrm{CF-A}} \;\ge\; m,
    \qquad
    q^\top z_t^+ - q^\top z_{\mathrm{CF-B}} \;\ge\; m.
    \label{eq:cf-margin-condition}
\end{equation}

\subsubsection{Counterfactual Ranking Objective}
\label{sec:method-cf-objective}
We optimize this margin constraint using a bi-directional hinge margin loss
$\mathcal{L}_{\mathrm{CF}}$:
\begin{equation}
    \mathcal{L}_{\mathrm{CF}} = \frac{1}{B} \sum_{i=1}^B \left[
        \max\left(0,\, m - q_i^\top z_{t,i}^+ + q_i^\top z_{\mathrm{CF-A},i}\right)
      + \max\left(0,\, m - q_i^\top z_{t,i}^+ + q_i^\top z_{\mathrm{CF-B},i}\right)
    \right],
    \label{eq:loss-cf}
\end{equation}
where $B = 16$ is the mini-batch size per device, $m = 0.2$ is the fixed margin,
and all embedding vectors are $\ell_2$-normalized prior to inner product calculation.

\subsubsection{Overall Training Objective}
\label{sec:method-overall-objective}
The primary Stage-1 retrieval alignment loss $\mathcal{L}_{\mathrm{ret}}$ is
computed via symmetric in-batch cross-entropy:
\begin{equation}
    \mathcal{L}_{\mathrm{ret}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\left(q_i^\top z_{t,i}^+ / \tau\right)}{\sum_{j=1}^B \exp\left(q_i^\top z_{t,j}^+ / \tau\right)},
    \label{eq:loss-ret}
\end{equation}
with temperature scale $\tau$ fixed to $1.0$ (matching the pre-trained CLIP logit scale).
The total multi-task objective is:
\begin{equation}
    \mathcal{L} = \mathcal{L}_{\mathrm{ret}}
                + \lambda_{\mathrm{fac}} \mathcal{L}_{\mathrm{fac}}
                + \lambda_{\mathrm{CF}} \mathcal{L}_{\mathrm{CF}}
                + \lambda_{\mathrm{ver}} \mathcal{L}_{\mathrm{ver}},
    \label{eq:loss-total}
\end{equation}
where $\lambda_{\mathrm{fac}}$, $\lambda_{\mathrm{CF}}$, and
$\lambda_{\mathrm{ver}}$ are non-negative loss scaling factors.
The model variants investigated in this work correspond to the following configurations:
\begin{itemize}
    \item \textbf{B1 (Factorization baseline):}
    $(\lambda_{\mathrm{fac}}, \lambda_{\mathrm{CF}}, \lambda_{\mathrm{ver}}) = (0.2,\, 0.0,\, 0.0)$.
    \item \textbf{B4 (Counterfactual ranking):}
    $(\lambda_{\mathrm{fac}}, \lambda_{\mathrm{CF}}, \lambda_{\mathrm{ver}}) = (0.2,\, 1.0,\, 0.0)$.
    \item \textbf{B5 (Verification extension):}
    $(\lambda_{\mathrm{fac}}, \lambda_{\mathrm{CF}}, \lambda_{\mathrm{ver}}) = (0.2,\, 1.0,\, 0.5)$.
\end{itemize}

% -----------------------------------------------------------------
% 3.5. Candidate-Level Constraint Verification
% -----------------------------------------------------------------
\subsection{Candidate-Level Constraint Verification}
\label{sec:method-verification}

\subsubsection{Constraint Verifier}
\label{sec:method-verifier-arch}
While Stage-1 retrieval provides fast linear cosine ranking across the entire
gallery, it cannot capture fine-grained, non-linear feature interactions between
a candidate and the query constraints.
To provide candidate-level verification, B5 introduces the \texttt{ConstraintVerifier}
module.
For any candidate image $I_c$ with normalized visual feature $z_c$, the verifier
forms an input vector $x_c \in \mathbb{R}^{4d}$ concatenating the candidate,
preservation, composed edit, and modification text representations:
\begin{equation}
    x_c = \left[z_c;\, p;\, \tilde{e};\, t_m\right] \in \mathbb{R}^{4d}.
    \label{eq:verifier-input}
\end{equation}
The vector $x_c$ is processed by a multi-layer perceptron:
\begin{equation}
    \mathbf{o}_c = W_{\mathrm{ver},2} \cdot \mathrm{Dropout}\left(\mathrm{GELU}\left(W_{\mathrm{ver},1} x_c + b_{\mathrm{ver},1}\right)\right) + b_{\mathrm{ver},2} \in \mathbb{R}^3,
    \label{eq:verifier-mlp}
\end{equation}
where $W_{\mathrm{ver},1} \in \mathbb{R}^{d_h \times 4d}$,
$b_{\mathrm{ver},1} \in \mathbb{R}^{d_h}$,
$W_{\mathrm{ver},2} \in \mathbb{R}^{3 \times d_h}$,
$b_{\mathrm{ver},2} \in \mathbb{R}^3$,
$d_h = 512$, and the dropout probability is set to $0.1$.
The output logits are converted into calibrated probabilities via the logistic sigmoid:
\begin{equation}
    \mathbf{p}(c) = \begin{pmatrix} p_P(c) \\ p_E(c) \\ p_V(c) \end{pmatrix} = \sigma\left(\mathbf{o}_c\right) \in (0, 1)^3,
    \label{eq:verifier-probs}
\end{equation}
representing preservation satisfaction $p_P(c)$, edit satisfaction $p_E(c)$, and
constraint violation $p_V(c)$.

\subsubsection{Verifier Training Targets}
\label{sec:method-verifier-targets}
The verifier is trained using independent Binary Cross-Entropy~(BCE) losses across
all three heads on the positive target and both counterfactual candidates:
\begin{equation}
    \mathcal{L}_{\mathrm{ver}} = \frac{1}{3} \sum_{k \in \{+,\, \mathrm{CF-A},\, \mathrm{CF-B}\}}
        \mathrm{BCE}\left(\mathbf{p}(I_k),\, \mathbf{y}(I_k)\right),
    \label{eq:loss-verifier}
\end{equation}
where $\mathbf{y}(I_k)$ are the binary ground-truth pseudo-labels defined in
Eq.~\eqref{eq:cf-labels}.

% -----------------------------------------------------------------
% 3.6. Verifier-Based Candidate Reranking
% -----------------------------------------------------------------
\subsection{Verifier-Based Candidate Reranking}
\label{sec:method-reranking}

\subsubsection{Stage-1 Candidate Selection}
\label{sec:method-stage1-selection}
Evaluating the non-linear verifier network across every image in large-scale galleries
(e.g., $|\mathcal{G}| = 2{,}297$ in CIRR) is computationally prohibitive at test time.
Therefore, inference adopts a two-stage retrieve-and-rerank paradigm:
\begin{enumerate}
    \item \textbf{Stage-1 Global Retrieval:} Query vector $q$ is scored against all
    candidates in $\mathcal{G}$ via $S_G(c) = q^\top z_c$.
    Candidates are sorted in descending order of similarity, and the top-$K$ candidates
    ($K \in \{50, 100\}$) are selected for Stage-2 reranking.
    \item \textbf{Stage-2 Reranking Window:} Candidates ranked beyond rank $K$ are left
    unaltered in their Stage-1 positions.
\end{enumerate}

\subsubsection{Verifier-Augmented Score}
\label{sec:method-verifier-score}
For each candidate $c$ within the top-$K$ pool, the final retrieval score combines
the Stage-1 similarity with the logarithmic constraint probabilities:
\begin{equation}
    S_{\mathrm{final}}(c) = S_G(c) + \alpha \log p_P(c) + \beta \log p_E(c) - \gamma \log p_V(c),
    \label{eq:method-rerank-score}
\end{equation}
where $\alpha, \beta, \gamma \ge 0$ are non-negative weighting coefficients.
Probabilities are clamped to $\epsilon = 10^{-7}$ to avoid numerical instability
under the logarithm.
The default full verifier score sets $(\alpha, \beta, \gamma) = (0.5,\, 0.5,\, 0.5)$.
The top-$K$ candidates are re-ordered according to $S_{\mathrm{final}}(c)$, after which
standard retrieval metrics (R@$k$, GR@$k$) are evaluated on the merged list.

\subsubsection{Diagnostic Evaluation}
\label{sec:method-diagnostic-eval}
To isolate whether the verifier provides an authentic, target-discriminative signal
independent of Stage-1 cosine similarity, we define an oracle diagnostic protocol on
CIRR validation.
We identify the subset of 491 queries whose ground-truth target image already ranks
within the Stage-1 top-50 candidates ($K = 50$).
Within this controlled 50-candidate pool, we evaluate candidate ranking under a pure
verifier-only scoring function without Stage-1 similarity:
\begin{equation}
    S_V(c) = \log p_P(c) + \log p_E(c) + \log\left(1 - p_V(c)\right).
    \label{eq:verifier-diagnostic-score}
\end{equation}
This formulation allows direct measurement of within-query pairwise win rates,
receiver operating characteristic area (AUROC), and mean target rank against Stage-1
and random baselines, isolating the intrinsic retrieval utility of the learned
constraint representations.

% =================================================================
% =================================================================
% SECTION 4: EXPERIMENTS
% =================================================================
% =================================================================
"""

# Read existing experiments.tex
tex_path = r'C:\New folder (3)\PE-CIR\paper\experiments.tex'
with open(tex_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the Section 4 start
target = r"""% Section counter set to 3 so \section{Experiments} numbers as 4
\setcounter{section}{3}

% =================================================================
\section{Experiments}
\label{sec:experiments}
% ================================================================="""

if target in content:
    new_content = content.replace(target, section3_text.strip() + "\n\n" + r"""\section{Experiments}
\label{sec:experiments}
% =================================================================""" )
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully replaced and inserted Section 3!")
else:
    print("Target string not found, searching alternative...")
    idx = content.find(r"\section{Experiments}")
    print("Found \\section{Experiments} at index:", idx)
