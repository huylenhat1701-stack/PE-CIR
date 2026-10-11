# -*- coding: utf-8 -*-
"""Script to expand Section 3 (Method) in paper/experiments.tex to comprehensive,
rigorous academic depth, preserving all 16 subsections without rambling or fabrication."""

import re

tex_path = r'C:\New folder (3)\PE-CIR\paper\experiments.tex'
with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Locate the beginning and end of current Section 3
sec3_marker = r"""% =================================================================
% =================================================================
% SECTION 3: METHOD
% =================================================================
% =================================================================
\section{Method}"""

sec4_marker = r"""% =================================================================
% =================================================================
% SECTION 4: EXPERIMENTS
% =================================================================
% =================================================================

\section{Experiments}"""

idx_sec3 = text.find(sec3_marker)
idx_sec4 = text.find(sec4_marker)

if idx_sec3 == -1 or idx_sec4 == -1:
    print(f"Error locating section markers: sec3={idx_sec3}, sec4={idx_sec4}")
    exit(1)

expanded_sec3 = r"""% =================================================================
% =================================================================
% SECTION 3: METHOD
% =================================================================
% =================================================================
\section{Method}
\label{sec:method}

In this section, we present the theoretical and architectural formulation of the
CC-CIR framework (\textbf{C}ounterfactual \textbf{C}onstraint learning for
\textbf{C}omposed \textbf{I}mage \textbf{R}etrieval).
We begin in Section~\ref{sec:method-formulation} by formalising the Composed Image
Retrieval task and dissecting the intrinsic tension between visual preservation
and modification constraints.
Section~\ref{sec:method-factorization} describes the preserve--edit factorization
mechanism, text-conditioned slot gating, and Stage-1 composed query construction
operating over a frozen vision--language backbone.
Section~\ref{sec:method-cf-construction} details the synthesis of structured
counterfactual hard negatives (CF-A and CF-B) from uncurated web captions.
Section~\ref{sec:method-cf-learning} defines the counterfactual margin ranking
objective and composite multi-task training loss.
Section~\ref{sec:method-verification} introduces the candidate-level constraint
verifier network and its binary cross-entropy supervision.
Finally, Section~\ref{sec:method-reranking} formulates the two-stage retrieve-and-rerank
pipeline, verifier-augmented scoring functions, and oracle diagnostic protocols.

% -----------------------------------------------------------------
% 3.1. Problem Formulation
% -----------------------------------------------------------------
\subsection{Problem Formulation}
\label{sec:method-formulation}

\subsubsection{Composed Image Retrieval}
\label{sec:method-cir}
Composed Image Retrieval~(CIR) represents an interactive multimodal retrieval paradigm
that extends conventional content-based image retrieval by allowing relative,
language-guided conditioning~\citep{liu2021cirr,wu2021fashioniq}.
Formally, let $\mathcal{I}$ denote the open-domain image space and $\mathcal{T}$ denote
the space of free-form natural language strings.
A user query is formulated as a multimodal composite pair $q = (I_r, T_m)$, where
$I_r \in \mathcal{I}$ serves as the visual reference image establishing the base context,
and $T_m \in \mathcal{T}$ represents a modification instruction describing the desired
semantic alterations relative to $I_r$.

Given an uncurated gallery of candidate images $\mathcal{G} = \{I_c\}_{c=1}^{|\mathcal{G}|}
\subset \mathcal{I}$, the objective of the retrieval system is to rank the gallery
candidates such that the ground-truth target image $I_t^* \in \mathcal{G}$ is positioned
at the top of the ranked list.
Unlike standard text-to-image (T2I) retrieval where the text query must describe the
entire visual scene from scratch, CIR requires the retrieval model to interpret $T_m$
as an edit operator applied directly to the visual semantics of $I_r$.
Retrieval efficacy is evaluated using Recall at rank $k$ (R@$k$), defined as the proportion
of queries for which the true target $I_t^*$ appears within the top-$k$ retrieved candidates:
\begin{equation}
    \mathrm{R@}k = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbf{1}\left[\mathrm{rank}(I_t^* \mid q) \le k\right],
    \label{eq:recall-def}
\end{equation}
where $\mathcal{Q}$ is the query evaluation set and $\mathrm{rank}(I_t^* \mid q) \in \{1, \dots, |\mathcal{G}|\}$
is the position of the target in the sorted gallery.
In benchmarks such as CIRR~\citep{liu2021cirr}, evaluation also incorporates Group Recall
(GR@$k$), which restricts the candidate set to a query-specific subset of six semantically
confusable distractor images $G_q \subset \mathcal{G}$ (excluding the reference $I_r$):
\begin{equation}
    \mathrm{GR@}k = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbf{1}\left[\mathrm{rank}_{G_q}(I_t^* \mid q) \le k\right].
    \label{eq:group-recall-def}
\end{equation}
Group Recall poses a strictly harder local discrimination challenge because the model
cannot rely on global semantic clustering to separate the target from visually distant
distractors.

\subsubsection{Preservation and Modification Constraints}
\label{sec:method-constraints}
A fundamental challenge in CIR arises from the fact that the textual instruction $T_m$
simultaneously imposes two conflicting semantic constraints on the retrieved candidate:
\begin{enumerate}
    \item \textbf{Preservation constraint ($\mathcal{C}_P$):} All visual attributes,
    contextual background elements, object geometry, and identity characteristics of
    $I_r$ that are not explicitly designated for alteration must remain strictly invariant
    in the retrieved candidate.
    Formally, let $\mathcal{A}(I)$ denote the complete set of visual attributes manifest
    in image $I$, and let $\mathcal{A}_{\mathrm{mod}} \subset \mathcal{A}(I_r)$ represent
    the specific attribute subset targeted for transformation by $T_m$.
    The preservation constraint requires that:
    \begin{equation}
        \mathcal{A}(I_t^*) \cap \left(\mathcal{A}(I_r) \setminus \mathcal{A}_{\mathrm{mod}}\right)
        \;=\; \mathcal{A}(I_r) \setminus \mathcal{A}_{\mathrm{mod}}.
        \label{eq:preservation-constraint}
    \end{equation}

    \item \textbf{Modification constraint ($\mathcal{C}_E$):} The semantic alterations
    dictated by $T_m$ must be faithfully incorporated into the target image.
    Letting $\mathcal{A}'_{\mathrm{mod}}$ denote the modified target attributes,
    the edit constraint requires:
    \begin{equation}
        \mathcal{A}'_{\mathrm{mod}} \;\subseteq\; \mathcal{A}(I_t^*).
        \label{eq:edit-constraint}
    \end{equation}
\end{enumerate}

These two constraints form an intrinsic Pareto tension in multimodal representation space.
Conventional composed retrieval architectures typically project $(I_r, T_m)$ into a single
monolithic query vector via unconstrained cross-attention or token injection, computing
an unweighted similarity score $S(c) = \mathrm{sim}(q, z_c)$.
Under such unconstrained optimization, models consistently suffer from two pervasive
failure modes:
\begin{itemize}
    \item \textbf{Reference bias (Preservation collapse):} The retrieval model over-indexes
    on visual similarity to the reference image $I_r$.
    As a consequence, the retrieved top candidates satisfy $\mathcal{C}_P$ but completely
    ignore the edit instruction $T_m$, effectively behaving as near-duplicate image search.
    \item \textbf{Edit bias (Identity collapse):} The retrieval model over-indexes on the
    linguistic tokens in $T_m$.
    As a consequence, the retrieved candidates depict objects exhibiting the modified
    attribute $\mathcal{A}'_{\mathrm{mod}}$ but belonging to an entirely different object
    category or visual scene, thereby violating $\mathcal{C}_P$.
\end{itemize}
Addressing CIR effectively necessitates an explicit structural decomposition that separates
$\mathcal{C}_P$ from $\mathcal{C}_E$ and verifies both constraints independently.

% -----------------------------------------------------------------
% 3.2. Preserve--Edit Factorization
% -----------------------------------------------------------------
\subsection{Preserve--Edit Factorization}
\label{sec:method-factorization}

\subsubsection{Frozen Vision--Language Backbone}
\label{sec:method-backbone}
To provide a rich, open-domain semantic foundation while avoiding catastrophic forgetting
and overfitting on small CIR training sets, we ground the entire framework upon a
pre-trained CLIP ViT-L-14-quickgelu~\citep{radford2021clip} backbone.
The visual encoder $E_{\mathrm{img}}: \mathcal{I} \to \mathbb{R}^d$ consists of a 24-layer
Vision Transformer operating on $14 \times 14$ non-overlapping image patches, producing
a global feature representation with latent dimension $d = 768$.
The text encoder $E_{\mathrm{txt}}: \mathcal{T} \to \mathbb{R}^d$ consists of a 12-layer
Transformer with masked causal self-attention operating over sub-word byte-pair encodings.

Crucially, all parameters of both $E_{\mathrm{img}}$ and $E_{\mathrm{txt}}$ are kept
strictly frozen throughout all training phases.
Freezing the backbone ensures three vital methodological advantages:
(i)~it preserves the zero-shot generalization capabilities acquired during web-scale
pre-training;
(ii)~it guarantees that differences in empirical retrieval performance stem strictly
from the learned factorization and constraint mechanisms rather than backbone fine-tuning;
and (iii)~it drastically reduces computational overhead during training.
For any input query pair $(I_r, T_m)$, we extract $\ell_2$-normalized global embeddings:
\begin{equation}
    z_r = \frac{E_{\mathrm{img}}(I_r)}{\|E_{\mathrm{img}}(I_r)\|_2} \in \mathbb{R}^d,
    \qquad
    t_m = \frac{E_{\mathrm{txt}}(T_m)}{\|E_{\mathrm{txt}}(T_m)\|_2} \in \mathbb{R}^d.
    \label{eq:backbone-embeddings}
\end{equation}
Each gallery candidate image $I_c \in \mathcal{G}$ is embedded offline as
$z_c = E_{\mathrm{img}}(I_c) / \|E_{\mathrm{img}}(I_c)\|_2 \in \mathbb{R}^d$.

\subsubsection{Preserve--Edit Factorization}
\label{sec:method-factorizer}
The \texttt{PreserveEditFactorizer} module decomposes the global reference embedding
$z_r$ into two disentangled representations conditioned on the modification instruction $t_m$:
a \emph{preservation representation} $p \in \mathbb{R}^d$ and an \emph{edit representation}
$e \in \mathbb{R}^d$.

First, the global visual feature vector $z_r$ is projected into $M = 8$ learnable semantic slots
$S \in \mathbb{R}^{M \times d}$.
We employ a linear slot projection matrix $W_{\mathrm{slot}} \in \mathbb{R}^{(M \cdot d) \times d}$,
followed by residual addition of the reference identity and layer normalization:
\begin{equation}
    S^{(0)} = \mathrm{reshape}\left(W_{\mathrm{slot}} z_r,\, (M, d)\right),
    \qquad
    S = \mathrm{LayerNorm}\left(S^{(0)} + \mathbf{1}_M z_r^\top\right),
    \label{eq:slot-expansion}
\end{equation}
where $\mathbf{1}_M \in \mathbb{R}^{M \times 1}$ is an all-ones vector broadcasting $z_r$
across each of the $M$ slots.
The residual connection ensures that all semantic slots inherit the base identity of $I_r$
prior to slot specialization.

Second, a text-conditioned gating mechanism dynamically scores the relevance of each
slot to the requested modification text.
For each slot index $i \in \{1, \dots, M\}$, intermediate gate hidden states
$H_i \in \mathbb{R}^{d_h}$ and scalar gating activations $g_i \in (0, 1)$ are computed via:
\begin{equation}
    H_i = \tanh\left(W_z S_i + W_t t_m\right),
    \qquad
    g_i = \sigma\left(w^\top H_i\right),
    \label{eq:gating}
\end{equation}
where $W_z \in \mathbb{R}^{d_h \times d}$, $W_t \in \mathbb{R}^{d_h \times d}$,
$w \in \mathbb{R}^{d_h}$, $d_h = 512$ is the hidden layer dimension, and $\sigma(\cdot)$
denotes the logistic sigmoid function.
The activation $g_i$ quantifies the probability that slot $S_i$ encodes semantics targeted
for modification by $T_m$.

Third, we compute the factorized edit representation $e$ and preserve representation $p$
via complementary weighted slot pooling:
\begin{equation}
    e = \frac{\sum_{i=1}^M g_i S_i}{\sum_{i=1}^M g_i + \epsilon},
    \qquad
    p = \frac{\sum_{i=1}^M (1 - g_i) S_i}{\sum_{i=1}^M (1 - g_i) + \epsilon},
    \label{eq:factorized-pooling}
\end{equation}
where $\epsilon = 10^{-7}$ provides numerical stability.
By construction, slots assigned high gating weights ($g_i \to 1$) aggregate into the edit
vector $e$, while slots with low gating weights ($1 - g_i \to 1$) aggregate into the
preserve vector $p$.

To enforce semantic disentanglement between $p$ and $e$ and prevent slot collapse, the
factorizer is trained with an auxiliary multi-term regularization loss $\mathcal{L}_{\mathrm{fac}}$:
\begin{equation}
    \mathcal{L}_{\mathrm{fac}} = \mathcal{L}_{\mathrm{leak}} + \mathcal{L}_{\mathrm{align}} + 0.1 \cdot \mathcal{L}_{\mathrm{balance}}.
    \label{eq:loss-fac}
\end{equation}
The constituent terms are defined as follows:
\begin{itemize}
    \item \textbf{Preservation leakage penalty ($\mathcal{L}_{\mathrm{leak}}$):}
    Penalizes cosine similarity between the preserve representation $p$ and the modification
    text $t_m$, ensuring that modification semantics do not leak into the preserved features:
    \begin{equation}
        \mathcal{L}_{\mathrm{leak}} = \left(\frac{p^\top t_m}{\|p\|_2 \|t_m\|_2}\right)^2.
        \label{eq:loss-leak}
    \end{equation}
    \item \textbf{Edit alignment loss ($\mathcal{L}_{\mathrm{align}}$):}
    Enforces cosine alignment between the composed edit vector $\tilde{e}$ (defined in
    Section~\ref{sec:method-stage1}) and the modification text embedding $t_m$:
    \begin{equation}
        \mathcal{L}_{\mathrm{align}} = 1 - \frac{\tilde{e}^\top t_m}{\|\tilde{e}\|_2 \|t_m\|_2}.
        \label{eq:loss-align}
    \end{equation}
    \item \textbf{Gate balance regularizer ($\mathcal{L}_{\mathrm{balance}}$):}
    Prevents degenerate solutions where the gating activations saturate uniformly towards
    all-zeros or all-ones across all slots:
    \begin{equation}
        \mathcal{L}_{\mathrm{balance}} = \left(\frac{1}{M}\sum_{i=1}^M g_i - 0.5\right)^2.
        \label{eq:loss-balance}
    \end{equation}
\end{itemize}

\subsubsection{Stage-1 Retrieval Representation}
\label{sec:method-stage1}
The \texttt{EditComposer} module merges the factorized edit slot $e$ with the raw text
embedding $t_m$ to construct the composed edit vector $\tilde{e} \in \mathbb{R}^d$:
\begin{equation}
    \tilde{e} = \mathrm{MLP}_{\mathrm{edit}}\left([e;\, t_m]\right),
    \label{eq:composed-edit}
\end{equation}
where $[\cdot; \cdot]$ denotes feature concatenation along the latent dimension, and
$\mathrm{MLP}_{\mathrm{edit}}$ comprises a $\text{Linear}(2d \to d_h)$ layer,
$\text{GELU}$ activation, and a $\text{Linear}(d_h \to d)$ projection ($d_h = 512$).

The final Stage-1 retrieval query representation $q \in \mathbb{R}^d$ is generated by
projecting and aggregating the preserve representation $p$, the composed edit $\tilde{e}$,
and the text embedding $t_m$:
\begin{equation}
    q = \mathrm{normalize}\left(W_p p + W_e \tilde{e} + W_m t_m\right),
    \label{eq:stage1-query}
\end{equation}
where $W_p, W_e, W_m \in \mathbb{R}^{d \times d}$ are learnable projection matrices, and
$\mathrm{normalize}(x) = x / \|x\|_2$.
The Stage-1 similarity score between query $q$ and any candidate $I_c$ with embedding
$z_c$ is evaluated via cosine similarity:
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
In standard contrastive learning frameworks for CIR, negative samples are drawn randomly
from within the mini-batch.
Random negatives almost invariably depict completely unrelated visual concepts: they share
neither the object class nor the scene context of $I_r$, and they fail to match the text $T_m$.
Consequently, random negatives simultaneously violate both the preservation constraint
$\mathcal{C}_P$ and the edit constraint $\mathcal{C}_E$ (denoted as $P^- E^-$ negatives).
Because $P^- E^-$ negatives are trivial to reject based on coarse categorical dissimilarities,
the model is never forced to learn the precise, orthogonal boundaries of $\mathcal{C}_P$
and $\mathcal{C}_E$.

To instill explicit sensitivity to both constraint axes, we introduce a structured
counterfactual synthesis pipeline that constructs two complementary hard negatives for
every query tuple:
\textbf{CF-A} ($P^+ E^-$), which satisfies preservation but violates the edit; and
\textbf{CF-B} ($P^- E^+$), which satisfies the edit but violates preservation.

\subsubsection{CF-A: Edit-Violation Counterfactual}
\label{sec:method-cf-a}
The CF-A candidate isolates the modification constraint $\mathcal{C}_E$ by explicitly
violating it while satisfying preservation $\mathcal{C}_P$.
CF-A depicts the exact same object entity or class category as the reference $I_r$ and
target $I_t^+$, but retains the pre-modification attribute value $a_{\mathrm{before}}$
instead of adopting the requested edit $a_{\mathrm{after}}$.
Formally, let $\mathcal{G}_{\mathrm{obj}(I_r)}$ be the subset of gallery candidates sharing
the object class of $I_r$.
The CF-A candidate is selected such that:
\begin{equation}
    I_{\mathrm{CF-A}} \in \mathcal{G}_{\mathrm{obj}(I_r)}, \quad
    a_{\mathrm{before}} \in \mathcal{A}(I_{\mathrm{CF-A}}), \quad
    I_{\mathrm{CF-A}} \neq I_t^+.
    \label{eq:cf-a-def}
\end{equation}
CF-A directly penalizes models that exhibit reference bias: if a model ignores the
modification text and relies solely on visual similarity to $I_r$, it will erroneously
rank $I_{\mathrm{CF-A}}$ higher than the true target $I_t^+$.

\subsubsection{CF-B: Preservation-Violation Counterfactual}
\label{sec:method-cf-b}
Conversely, the CF-B candidate isolates the preservation constraint $\mathcal{C}_P$
by violating it while satisfying the edit constraint $\mathcal{C}_E$.
CF-B incorporates the modified attribute $a_{\mathrm{after}}$, but manifests this attribute
on an entirely different object category or unrelated scene context.
Formally, letting $\mathcal{G}_{a_{\mathrm{after}}}$ denote candidates exhibiting the target
attribute $a_{\mathrm{after}}$, CF-B is selected such that:
\begin{equation}
    I_{\mathrm{CF-B}} \in \mathcal{G}_{a_{\mathrm{after}}}, \quad
    \mathrm{obj}(I_{\mathrm{CF-B}}) \neq \mathrm{obj}(I_r).
    \label{eq:cf-b-def}
\end{equation}
CF-B directly penalizes models that exhibit edit bias: if a model ignores reference image
context and matches candidates purely based on the linguistic terms in $T_m$, it will
erroneously rank $I_{\mathrm{CF-B}}$ higher than the correct target $I_t^+$.

\subsubsection{Counterfactual Pseudo-Labels}
\label{sec:method-cf-labels}
We mine training tuples from Conceptual Captions 3M (CC3M)~\citep{changpinyo2021cc12m}.
A lexical pattern-matching engine scans image captions against a curated dictionary of
37 visual object categories (e.g., \emph{dress, shirt, car, dog, chair, boat, bottle})
and four attribute families:
\begin{itemize}
    \item \textbf{Colour (12 values):} \emph{black, white, red, blue, green, yellow, brown, pink, purple, orange, gray, grey}.
    \item \textbf{Pattern (7 values):} \emph{plain, striped, floral, plaid, checked, dotted, printed}.
    \item \textbf{Material (9 values):} \emph{wooden, metal, leather, cotton, silk, denim, glass, plastic, wool}.
    \item \textbf{Size / Shape (9 values):} \emph{small, large, long, short, round, square, wide, narrow, tall}.
\end{itemize}
Scanning approximately 100{,}000 captioned records yielded 50{,}000 candidate tuples,
from which a finalized manifest of 45{,}000 tuples across 13{,}582 unique image paths
was constructed.
Each training instance forms a five-tuple:
\begin{equation}
    \tau = \left(I_r,\, I_t^+,\, I_{\mathrm{CF-A}},\, I_{\mathrm{CF-B}},\, T_m\right).
    \label{eq:tuple-def}
\end{equation}

For candidate-level constraint verification, we assign each candidate image a three-dimensional
binary ground-truth vector $\mathbf{y} = (y_P, y_E, y_V)^\top \in \{0, 1\}^3$ reflecting
preservation satisfaction ($y_P$), edit satisfaction ($y_E$), and constraint violation ($y_V$):
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
In Stage-1 embedding space, the composed query representation $q$ must enforce a clear
geometric separation: the cosine similarity of the true target $I_t^+$ must exceed that of
both CF-A and CF-B by at least a predefined margin $m > 0$:
\begin{equation}
    q^\top z_t^+ - q^\top z_{\mathrm{CF-A}} \;\ge\; m,
    \qquad
    q^\top z_t^+ - q^\top z_{\mathrm{CF-B}} \;\ge\; m.
    \label{eq:cf-margin-condition}
\end{equation}

\subsubsection{Counterfactual Ranking Objective}
\label{sec:method-cf-objective}
We optimize the separation condition in Eq.~\eqref{eq:cf-margin-condition} using a
bi-directional hinge margin loss $\mathcal{L}_{\mathrm{CF}}$:
\begin{equation}
    \mathcal{L}_{\mathrm{CF}} = \frac{1}{B} \sum_{i=1}^B \left[
        \max\left(0,\, m - q_i^\top z_{t,i}^+ + q_i^\top z_{\mathrm{CF-A},i}\right)
      + \max\left(0,\, m - q_i^\top z_{t,i}^+ + q_i^\top z_{\mathrm{CF-B},i}\right)
    \right],
    \label{eq:loss-cf}
\end{equation}
where $B = 16$ is the mini-batch size per device, $m = 0.2$ is the fixed ranking margin,
and all embeddings are $\ell_2$-normalized prior to inner product evaluation.
The hinge formulation ensures that gradients are only propagated for active violations
where either counterfactual encroaches within the safety margin $m$ of the positive candidate.

\subsubsection{Overall Training Objective}
\label{sec:method-overall-objective}
The global Stage-1 retrieval alignment loss $\mathcal{L}_{\mathrm{ret}}$ is computed via
in-batch symmetric cross-entropy over query and positive target pairs:
\begin{equation}
    \mathcal{L}_{\mathrm{ret}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\left(q_i^\top z_{t,i}^+ / \tau\right)}{\sum_{j=1}^B \exp\left(q_i^\top z_{t,j}^+ / \tau\right)},
    \label{eq:loss-ret}
\end{equation}
where the logit scale temperature is fixed to $\tau = 1.0$.

The composite multi-task objective combines the retrieval alignment loss, factorization loss,
counterfactual ranking loss, and constraint verification loss:
\begin{equation}
    \mathcal{L} = \mathcal{L}_{\mathrm{ret}}
                + \lambda_{\mathrm{fac}} \mathcal{L}_{\mathrm{fac}}
                + \lambda_{\mathrm{CF}} \mathcal{L}_{\mathrm{CF}}
                + \lambda_{\mathrm{ver}} \mathcal{L}_{\mathrm{ver}},
    \label{eq:loss-total}
\end{equation}
where $\lambda_{\mathrm{fac}}$, $\lambda_{\mathrm{CF}}$, and $\lambda_{\mathrm{ver}}$ are
loss scaling hyperparameters.
The three model variants investigated in this study correspond to the following configurations:
\begin{itemize}
    \item \textbf{B1 (Factorization baseline):}
    $(\lambda_{\mathrm{fac}}, \lambda_{\mathrm{CF}}, \lambda_{\mathrm{ver}}) = (0.2,\, 0.0,\, 0.0)$.
    \item \textbf{B4 (Counterfactual ranking):}
    $(\lambda_{\mathrm{fac}}, \lambda_{\mathrm{CF}}, \lambda_{\mathrm{ver}}) = (0.2,\, 1.0,\, 0.0)$.
    \item \textbf{B5 (Verification extension):}
    $(\lambda_{\mathrm{fac}}, \lambda_{\mathrm{CF}}, \lambda_{\mathrm{ver}}) = (0.2,\, 1.0,\, 0.5)$.
\end{itemize}
All variants are optimized using AdamW~\citep{loshchilov2018decoupled} with learning rate $10^{-4}$,
weight decay $0.1$, batch size 16, and five epochs under random seed 0.
B5 is initialized afresh from random weights rather than warm-started from B4.

% -----------------------------------------------------------------
% 3.5. Candidate-Level Constraint Verification
% -----------------------------------------------------------------
\subsection{Candidate-Level Constraint Verification}
\label{sec:method-verification}

\subsubsection{Constraint Verifier}
\label{sec:method-verifier-arch}
While Stage-1 retrieval relies on inner-product similarity in a shared latent space,
inner products are strictly bilinear and cannot capture non-linear, high-order semantic
interactions between candidate images and individual constraint components.
To enable fine-grained, non-linear verification, B5 introduces the \texttt{ConstraintVerifier}
module.

For any candidate image $I_c$ with normalized visual embedding $z_c$, the verifier
constructs a $4d$-dimensional input representation $x_c \in \mathbb{R}^{4d}$ concatenating
the candidate image, the factorized preservation feature $p$, the composed edit feature $\tilde{e}$,
and the modification text embedding $t_m$:
\begin{equation}
    x_c = \left[z_c;\, p;\, \tilde{e};\, t_m\right] \in \mathbb{R}^{4d}.
    \label{eq:verifier-input}
\end{equation}
The concatenated feature $x_c$ is passed through a multi-layer perceptron:
\begin{equation}
    \mathbf{o}_c = W_{\mathrm{ver},2} \cdot \mathrm{Dropout}\left(\mathrm{GELU}\left(W_{\mathrm{ver},1} x_c + b_{\mathrm{ver},1}\right)\right) + b_{\mathrm{ver},2} \in \mathbb{R}^3,
    \label{eq:verifier-mlp}
\end{equation}
where $W_{\mathrm{ver},1} \in \mathbb{R}^{d_h \times 4d}$,
$b_{\mathrm{ver},1} \in \mathbb{R}^{d_h}$,
$W_{\mathrm{ver},2} \in \mathbb{R}^{3 \times d_h}$,
$b_{\mathrm{ver},2} \in \mathbb{R}^3$,
$d_h = 512$, and the dropout probability is set to $0.1$.
The output logits $\mathbf{o}_c$ are mapped to calibrated probabilities via the element-wise
logistic sigmoid:
\begin{equation}
    \mathbf{p}(c) = \begin{pmatrix} p_P(c) \\ p_E(c) \\ p_V(c) \end{pmatrix} = \sigma\left(\mathbf{o}_c\right) \in (0, 1)^3,
    \label{eq:verifier-probs}
\end{equation}
representing preservation satisfaction probability $p_P(c)$, edit satisfaction probability
$p_E(c)$, and constraint violation probability $p_V(c)$.

\subsubsection{Verifier Training Targets}
\label{sec:method-verifier-targets}
The verifier network is trained using independent Binary Cross-Entropy~(BCE) losses across
all three heads on the positive target and both counterfactual candidates:
\begin{equation}
    \mathcal{L}_{\mathrm{ver}} = \frac{1}{3} \sum_{k \in \{+,\, \mathrm{CF-A},\, \mathrm{CF-B}\}}
        \mathrm{BCE}\left(\mathbf{p}(I_k),\, \mathbf{y}(I_k)\right),
    \label{eq:loss-verifier}
\end{equation}
where $\mathbf{y}(I_k)$ are the binary ground-truth pseudo-labels defined in Eq.~\eqref{eq:cf-labels}.
Evaluating $\mathcal{L}_{\mathrm{ver}}$ across all three candidates per tuple ensures that
each head receives balanced positive and negative supervision during every training iteration.

% -----------------------------------------------------------------
% 3.6. Verifier-Based Candidate Reranking
% -----------------------------------------------------------------
\subsection{Verifier-Based Candidate Reranking}
\label{sec:method-reranking}

\subsubsection{Stage-1 Candidate Selection}
\label{sec:method-stage1-selection}
Evaluating the non-linear verifier network across every image in open-domain search galleries
(e.g., $|\mathcal{G}| = 2{,}297$ in CIRR, or $|\mathcal{G}| > 10^5$ in web galleries) is
computationally prohibitive at query time.
Therefore, inference follows a scalable two-stage retrieve-and-rerank paradigm:
\begin{enumerate}
    \item \textbf{Stage-1 Fast Candidate Retrieval:} The composed query vector $q$ is scored
    against all gallery images $I_c \in \mathcal{G}$ via cosine similarity $S_G(c) = q^\top z_c$.
    Candidates are sorted in descending order of similarity, and the top-$K$ candidates
    ($K \in \{50, 100\}$) are selected for Stage-2 verifier reranking.
    \item \textbf{Stage-2 Reranking Window:} Candidates ranked beyond position $K$ remain
    in their original Stage-1 positions.
\end{enumerate}

\subsubsection{Verifier-Augmented Score}
\label{sec:method-verifier-score}
For each candidate $c$ within the top-$K$ pool, the final retrieval score combines the
Stage-1 cosine similarity with the logarithmic constraint probabilities:
\begin{equation}
    S_{\mathrm{final}}(c) = S_G(c) + \alpha \log p_P(c) + \beta \log p_E(c) - \gamma \log p_V(c),
    \label{eq:rerank-general}
\end{equation}
where $\alpha, \beta, \gamma \ge 0$ are non-negative weighting coefficients.
Probabilities are clamped from below to $\epsilon = 10^{-7}$ prior to taking logarithms
to prevent numerical instability.
The default full verifier scoring configuration sets $(\alpha, \beta, \gamma) = (0.5,\, 0.5,\, 0.5)$.
The top-$K$ candidates are re-sorted in descending order of $S_{\mathrm{final}}(c)$ and
merged with the remaining candidates, after which standard retrieval metrics are evaluated.

\subsubsection{Diagnostic Evaluation}
\label{sec:method-diagnostic-eval}
To isolate whether the verifier provides an authentic, target-discriminative retrieval
signal independent of Stage-1 cosine similarity, we define an oracle diagnostic protocol
on the CIRR validation split.
We filter the 4{,}181 CIRR validation queries to the subset of 491 queries whose ground-truth
target image already appears within the Stage-1 top-50 candidates ($K = 50$).
Within each of these 491 queries, we evaluate candidate ranking under a pure verifier-only
score without Stage-1 similarity:
\begin{equation}
    S_V(c) = \log p_P(c) + \log p_E(c) + \log\left(1 - p_V(c)\right).
    \label{eq:verifier-diagnostic-score}
\end{equation}
This formulation completely decouples the verifier from the Stage-1 backbone representation,
allowing direct measurement of within-query pairwise win rates, area under the ROC curve
(AUROC), and mean target rank against uniform-random and Stage-1 baselines.

"""

new_text = text[:idx_sec3] + expanded_sec3 + text[idx_sec4:]

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Section 3 expanded successfully!")
