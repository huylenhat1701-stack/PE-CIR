import re
from pathlib import Path

# Load source from paper/from_zip/experiments.tex
source_path = Path("paper/from_zip/experiments.tex")
target_path = Path("paper/experiments.tex")

with open(source_path, "r", encoding="utf-8") as f:
    text = f.read()

original_len = len(text)
print(f"Loaded {original_len} characters from {source_path}")

# 1. Replace "disentangled" in Abstract
s_old_abs = "First, a preserve/edit factorizer decomposes each modification text\ninto two disentangled semantic slots, producing a composed Stage-1\nretrieval query."
s_new_abs = "First, a preserve/edit factorizer decomposes reference visual semantics\ninto two complementary factorized semantic slots conditioned on the modification text,\nproducing a composed Stage-1 retrieval query."
assert s_old_abs in text, "Abstract snippet not found"
text = text.replace(s_old_abs, s_new_abs)

# 2. Replace "disentangling" in Introduction
s_old_intro1 = "model can learn to avoid it without explicitly disentangling the two\nconstraints."
s_new_intro1 = "model can learn to avoid it without explicitly factorizing the two\nconstraints."
assert s_old_intro1 in text, "Intro 1 snippet not found"
text = text.replace(s_old_intro1, s_new_intro1)

s_old_intro2 = "The core is a \\emph{preserve/edit factorizer} that projects the\nmodification text into two disentangled slots, enabling the Stage-1\nretrieval query to separately encode what must be preserved and what\nmust change."
s_new_intro2 = "The core is a \\emph{preserve/edit factorizer} that decomposes\nreference visual semantics into two complementary factorized slots conditioned on the\nmodification text, enabling the Stage-1 retrieval query to separately encode what must\nbe preserved and what must change."
assert s_old_intro2 in text, "Intro 2 snippet not found"
text = text.replace(s_old_intro2, s_new_intro2)

# 3. Update Section 3 roadmap
s_old_roadmap = """The section is organised as follows:
Section~\\ref{sec:method-formulation} formalises the Composed Image Retrieval task
and establishes the mathematical definitions of visual preservation ($\mathcal{C}_P$)
and modification ($\mathcal{C}_E$) constraints.
Section~\\ref{sec:method-factorization} describes the frozen vision--language backbone,
the learnable slot-based preserve--edit factorizer, and Stage-1 composed query synthesis.
Section~\\ref{sec:method-cf-construction} details the synthesis principles and lexical
mining protocol for structured counterfactual hard negatives (CF-A and CF-B).
Section~\\ref{sec:method-cf-learning} formulates the counterfactual margin ranking loss,
gradient dynamics, and global composite multi-task training objectives.
Section~\\ref{sec:method-verification} introduces the architectural design and binary
cross-entropy supervision of the candidate-level constraint verifier.
Finally, Section~\\ref{sec:method-reranking} formulates the two-stage retrieve-and-rerank
pipeline, verifier-augmented scoring mechanisms, and oracle diagnostic evaluation protocols."""

s_new_roadmap = """The section is organised as follows:
Section~\\ref{sec:method-formulation} formalises the Composed Image Retrieval task
and establishes the operational definitions of visual preservation ($\mathcal{C}_P$)
and modification ($\mathcal{C}_E$) constraints.
Section~\\ref{sec:method-factorization} describes the frozen vision--language backbone,
the learnable slot-based preserve--edit factorizer, and Stage-1 composed query synthesis.
Section~\\ref{sec:method-cf-construction} details the rule-based lexical mining protocol
for structured pseudo-counterfactual hard negatives (CF-A and CF-B).
Section~\\ref{sec:method-cf-learning} formulates the counterfactual margin ranking loss
and global composite multi-task training objectives.
Section~\\ref{sec:method-verification} introduces the architectural design and binary
cross-entropy supervision of the candidate-level constraint verifier.
Section~\\ref{sec:method-reranking} formulates the two-stage retrieve-and-rerank
pipeline, verifier-augmented scoring mechanisms, and oracle diagnostic evaluation protocols.
Section~\\ref{sec:method-algorithms} provides algorithmic workflows for multi-task
training and two-stage inference.
Section~\\ref{sec:method-mining} details the rule-based lexical grammar and mining pipeline.
Finally, Section~\\ref{sec:method-complexity} analyses computational complexity and runtime scalability."""

assert s_old_roadmap in text, "Roadmap snippet not found"
text = text.replace(s_old_roadmap, s_new_roadmap)

# 4. Soften C_P and C_E definitions in Section 3.1.2 using slice
cp_start = text.find(r"\subsubsection{Preservation and Modification Constraints}")
cp_end = text.find(r"% 3.2. Preserve--Edit Factorization")
assert cp_start != -1 and cp_end != -1, "C_P and C_E slice not found"

s_new_cp_ce = """\\subsubsection{Preservation and Modification Constraints}
\\label{sec:method-constraints}
A composed query $q = (I_r, T_m)$ inherently imposes two distinct, mutually complementary
semantic constraints on any candidate image $I_c \\in \\mathcal{G}$:
\\begin{enumerate}
    \\item \\textbf{Preservation constraint ($\mathcal{C}_P$):} The visual identity,
    primary object category, and unedited contextual elements manifest in the reference image $I_r$
    that are not explicitly designated for alteration by $T_m$ should remain preserved in the target image.
    Conceptually, if $\\mathcal{A}(I)$ denotes the attribute set of image $I$,
    and $\\mathcal{A}_{\\mathrm{mod}} \\subset \\mathcal{A}(I_r)$ denotes the specific
    attribute subset targeted for modification by $T_m$, an idealized preservation condition
    aims for invariance over the complementary attribute set:
    \\begin{equation}
        \\mathcal{A}(I_t^*) \\cap \\left(\\mathcal{A}(I_r) \\setminus \\mathcal{A}_{\\mathrm{mod}}\\right)
        \\;\\approx\\; \\mathcal{A}(I_r) \\setminus \\mathcal{A}_{\\mathrm{mod}}.
        \\label{eq:preservation-constraint-formal}
    \\end{equation}

    \\item \\textbf{Modification constraint ($\mathcal{C}_E$):} The specific semantic
    transformations dictated by $T_m$ must be faithfully incorporated into the target image.
    Letting $\\mathcal{A}'_{\\mathrm{mod}}$ denote the modified attribute values specified
    by the edit instruction, an idealized modification condition requires:
    \\begin{equation}
        \\mathcal{A}'_{\\mathrm{mod}} \\;\\subseteq\\; \\mathcal{A}(I_t^*).
        \\label{eq:edit-constraint-formal}
    \\end{equation}
\\end{enumerate}

\\paragraph{Operational Reality in Open-Domain Retrieval.}
We note that Equations~\\eqref{eq:preservation-constraint-formal} and~\\eqref{eq:edit-constraint-formal}
represent formal conceptual ideals rather than strict discrete guarantees.
In real-world CIR datasets (e.g., CIRR, Fashion-IQ) and web-scale image collections (e.g., CC3M),
images naturally exhibit continuous variations in camera viewpoint, illumination, background
clutter, and unannotated contextual details.
Consequently, in the PE-CIR framework:
\\begin{itemize}
    \\item $\\mathcal{C}_P$ is operationalized as preserving the primary object class, category semantics,
    and non-target visual characteristics of the reference image $I_r$.
    \\item $\\mathcal{C}_E$ is operationalized as satisfying the requested semantic transformation
    expressed in $T_m$ (e.g., transitioning an attribute from a source value to a target value).
\\end{itemize}

In practice, $\\mathcal{C}_P$ and $\\mathcal{C}_E$ define a Pareto tension in multimodal
feature space.
Conventional composed retrieval approaches typically map $(I_r, T_m)$ into a single
monolithic latent vector $q \\in \\mathbb{R}^d$ via unconstrained fusion (e.g., cross-attention,
transformer concatenation, or token injection), ranking gallery candidates by raw cosine
similarity $S_G(c) = \\cos(q, z_c)$.
Under unconstrained optimization, models consistently collapse into one of two systematic
pathologies:
\\begin{itemize}
    \\item \\textbf{Reference bias (Preservation collapse):} The query representation is
    dominated by the visual features of $I_r$.
    The model retrieves candidates that satisfy $\\mathcal{C}_P$ with high fidelity but
    completely ignore the modification text $T_m$, effectively collapsing CIR into near-duplicate
    image retrieval ($P^+ E^-$ candidates).
    \\item \\textbf{Edit bias (Identity collapse):} The query representation is dominated
    by the lexical tokens of $T_m$.
    The model retrieves candidates that display the requested target attribute
    $\\mathcal{A}'_{\\mathrm{mod}}$ but belong to a completely different object category or
    unrelated scene context, thus violating $\\mathcal{C}_P$ ($P^- E^+$ candidates).
\\end{itemize}
Overcoming these pathologies requires an architecture that explicitly decomposes
$\\mathcal{C}_P$ from $\\mathcal{C}_E$ into complementary factorized representations during
learning and verifies both constraints during candidate scoring.

% -----------------------------------------------------------------
"""
text = text[:cp_start] + s_new_cp_ce + text[cp_end:]

# 5. Section 3.2.2: replace orthogonal and disentanglement
s_old_fac1 = "orthogonal, text-conditioned representations: a preserve vector $p \\in \\mathbb{R}^d$ and\nan edit vector $e \\in \\mathbb{R}^d$."
s_new_fac1 = "complementary, text-conditioned representations: a preserve vector $p \\in \\mathbb{R}^d$ and\nan edit vector $e \\in \\mathbb{R}^d$."
assert s_old_fac1 in text, "Factorizer 1 snippet not found"
text = text.replace(s_old_fac1, s_new_fac1)

s_old_fac2 = "To enforce semantic disentanglement, we supervise the factorizer with an auxiliary\nregularization loss $\\mathcal{L}_{\\mathrm{fac}}$"
s_new_fac2 = "To encourage semantic factorization and minimize leakage, we supervise the factorizer with an auxiliary\nregularization loss $\\mathcal{L}_{\\mathrm{fac}}$"
assert s_old_fac2 in text, "Factorizer 2 snippet not found"
text = text.replace(s_old_fac2, s_new_fac2)

# 6. Section 3.3.1 - 3.3.3: Ground CF-A and CF-B in rule-based caption mining via slice
cf_start = text.find(r"\subsection{Counterfactual Constraint Construction}")
cf_end = text.find(r"\subsubsection{Counterfactual Pseudo-Labels}")
assert cf_start != -1 and cf_end != -1, "CF slice not found"

s_new_cf = """\\subsection{Counterfactual Constraint Construction}
\\label{sec:method-cf-construction}

\\subsubsection{Motivation}
\\label{sec:method-cf-motivation}
In standard contrastive learning frameworks for CIR, negative samples are drawn randomly
from within the mini-batch.
Random negatives almost invariably depict completely unrelated visual entities: they share
neither the object class nor the visual context of $I_r$, and they fail to match the text $T_m$.
Consequently, random negatives simultaneously violate both the preservation constraint
$\\mathcal{C}_P$ and the edit constraint $\\mathcal{C}_E$ (denoted as $P^- E^-$ negatives).
Because $P^- E^-$ negatives are trivial to reject based on coarse categorical dissimilarities,
the model is never forced to learn the complementary boundaries of $\\mathcal{C}_P$
and $\\mathcal{C}_E$.

To instill explicit sensitivity to both constraint axes, we introduce a structured
counterfactual mining pipeline that constructs two complementary hard negatives for
every query tuple via rule-based lexical matching over CC3M captions:
\\textbf{CF-A} ($P^+ E^-$), which satisfies preservation but violates the edit; and
\\textbf{CF-B} ($P^- E^+$), which satisfies the edit but violates preservation.

\\subsubsection{CF-A: Edit-Violation Counterfactual}
\\label{sec:method-cf-a}
The CF-A candidate isolates the modification constraint $\\mathcal{C}_E$ by serving as an
operational proxy for an edit-violation negative ($P^+ E^-$).
In our rule-based mining pipeline (detailed in Section~\\ref{sec:method-mining}), CF-A is
selected from the training corpus such that its caption matches the same object category
keyword as the reference $I_r$, but retains the pre-modification attribute value $a_{\\mathrm{before}}$
instead of adopting the requested edit $a_{\\mathrm{after}}$.
Formally, let $\\mathcal{G}_{\\mathrm{obj}(I_r)}$ be the subset of candidate records sharing
the detected object class of $I_r$.
The CF-A candidate is mined such that:
\\begin{equation}
    I_{\\mathrm{CF-A}} \\in \\mathcal{G}_{\\mathrm{obj}(I_r)}, \\quad
    a_{\\mathrm{before}} \\in \\mathcal{A}_{\\mathrm{caption}}(I_{\\mathrm{CF-A}}), \\quad
    I_{\\mathrm{CF-A}} \\neq I_t^+.
    \\label{eq:cf-a-formal-def}
\\end{equation}
CF-A directly penalizes models that exhibit reference bias: if a model ignores the
modification text and relies solely on visual similarity to $I_r$, it will erroneously
rank $I_{\\mathrm{CF-A}}$ higher than the true target $I_t^+$.
Because CF-A is mined via caption keyword matching on uncurated web images rather than
generated via controlled image editing, it serves as a practical lexical proxy for
edit omission under natural web visual diversity.

\\subsubsection{CF-B: Preservation-Violation Counterfactual}
\\label{sec:method-cf-b}
Conversely, the CF-B candidate isolates the preservation constraint $\\mathcal{C}_P$ by
serving as an operational proxy for a preservation-violation negative ($P^- E^+$).
CF-B incorporates the modified attribute $a_{\\mathrm{after}}$, but manifests this attribute
on an entirely different object category or unrelated scene context.
Formally, letting $\\mathcal{G}_{a_{\\mathrm{after}}}$ denote candidate records exhibiting the
target attribute keyword $a_{\\mathrm{after}}$ in their caption, CF-B is mined such that:
\\begin{equation}
    I_{\\mathrm{CF-B}} \\in \\mathcal{G}_{a_{\\mathrm{after}}}, \\quad
    \\mathrm{obj}(I_{\\mathrm{CF-B}}) \\neq \\mathrm{obj}(I_r).
    \\label{eq:cf-b-formal-def}
\\end{equation}
CF-B directly penalizes models that exhibit edit bias: if a model ignores reference image
context and matches candidates purely based on the linguistic terms in $T_m$, it will
erroneously rank $I_{\\mathrm{CF-B}}$ higher than the correct target $I_t^+$.
Like CF-A, CF-B is a rule-based lexical proxy derived from web captions, subject to the
natural background and stylistic variations present in web-scraped data.

"""
text = text[:cf_start] + s_new_cf + text[cf_end:]

# 7. Correct Eq. (24) description in Section 3.4.3: symmetric -> in-batch query-to-target cross-entropy
s_old_eq24 = """\\subsubsection{Overall Training Objective}
\\label{sec:method-overall-objective}
The global Stage-1 retrieval alignment loss $\\mathcal{L}_{\\mathrm{ret}}$ is computed via
in-batch symmetric cross-entropy over query and positive target pairs:
\\begin{equation}
    \\mathcal{L}_{\\mathrm{ret}} = -\\frac{1}{B} \\sum_{i=1}^B \\log \\frac{\\exp\\left(q_i^\\top z_{t,i}^+ / \\tau\\right)}{\\sum_{j=1}^B \\exp\\left(q_i^\\top z_{t,j}^+ / \\tau\\right)},
    \\label{eq:retrieval-alignment-loss-eq}
\\end{equation}
where the logit scale temperature is fixed to $\\tau = 1.0$ (matching the pre-trained CLIP
logit scale)."""

s_new_eq24 = """\\subsubsection{Overall Training Objective}
\\label{sec:method-overall-objective}
The global Stage-1 retrieval alignment loss $\\mathcal{L}_{\\mathrm{ret}}$ is computed as an
in-batch query-to-target cross-entropy loss over query and positive target pairs:
\\begin{equation}
    \\mathcal{L}_{\\mathrm{ret}} = -\\frac{1}{B} \\sum_{i=1}^B \\log \\frac{\\exp\\left(q_i^\\top z_{t,i}^+ / \\tau\\right)}{\\sum_{j=1}^B \\exp\\left(q_i^\\top z_{t,j}^+ / \\tau\\right)},
    \\label{eq:retrieval-alignment-loss-eq}
\\end{equation}
where the logit scale temperature is fixed to $\\tau = 1.0$ (matching the pre-trained CLIP
logit scale).
Unlike symmetric bidirectional contrastive objectives used during vision--language pre-training,
Stage-1 CIR alignment optimizes the query-to-candidate retrieval distribution directly,
aligning precisely with the inference retrieval ranking direction as implemented in
\\texttt{retrieval\\_alignment\\_loss}."""

assert s_old_eq24 in text, "Eq 24 snippet not found"
text = text.replace(s_old_eq24, s_new_eq24)

# 8. Delete Sections 3.9, 3.10, and 3.11, renumber 3.12 to 3.9
s_target_start = "% 3.9. Architectural Evolution: B0 to B5"
s_target_end = "% 3.12. Computational Complexity and Scalability"

idx_start = text.find(s_target_start)
idx_end = text.find(s_target_end)

assert idx_start != -1 and idx_end != -1, "Sections 3.9-3.11 markers not found"

replacement_heading = "% 3.9. Computational Complexity and Scalability"
text = text[:idx_start] + replacement_heading + text[idx_end + len(s_target_end):]

# 9. In Complexity section: remove "edge memory" claim
s_old_edge = "allowing the entire retrieval system to reside comfortably in low-end GPU or edge memory."
s_new_edge = "allowing the entire retrieval index and model parameters to be served efficiently in GPU memory on standard evaluation and retrieval servers without prohibitive resource demands."
assert s_old_edge in text, "Edge memory snippet not found"
text = text.replace(s_old_edge, s_new_edge)

# 10. In Section 4.1: replace "two disentangled semantic components"
s_old_sec4_dis = "modification text into two disentangled semantic components."
s_new_sec4_dis = "modification text into two complementary factorized semantic components."
assert s_old_sec4_dis in text, "Section 4.1 disentangled snippet not found"
text = text.replace(s_old_sec4_dis, s_new_sec4_dis)

# Write to target_path
with open(target_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Successfully wrote {len(text)} characters to {target_path}")
