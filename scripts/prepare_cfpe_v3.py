"""Object-type preservation pilot from single-attribute CC3M noun phrases.

This changes the preservation task relative to strict v2. Positive/CF-A have
the reference object type; CF-B has a different object type and the desired
edited attribute. These caption-derived labels require visual audit.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict

from prepare_cfpe_v2 import KINDS, OBJECTS, ROLES, TOKEN_KIND, VOCAB, main, normalize

NOUNS = OBJECTS | set("tulip rose lily iris mushroom violin earring sleigh turtle door bucket ruler ball sphere piano guitar train airplane helicopter bicycle bicycle helmet umbrella suitcase backpack skirt sweater blouse sock glove scarf bench desk cabinet shelf vase bowl mug watch clock camera laptop keyboard telephone refrigerator oven bicycle tractor van ship yacht canoe apple banana lemon orange cookie cake sandwich pizza burger horse elephant zebra tiger lion bear rabbit frog deer fox wolf butterfly fish dolphin whale penguin duck chicken eagle parrot snake lizard cow sheep goat pig nail".split())
ALIASES = {"puppy": "dog", "puppies": "dog", "kitten": "cat", "kittens": "cat",
           "bike": "bicycle", "airplane": "plane", "aeroplane": "plane",
           "automobile": "car", "sofa": "couch", "couch": "couch",
           "grey": "gray"}
# Keep alternatives such as puppy/dog and bike/bicycle out of false CF-B pairs.
NOUNS |= {"puppy", "kitten", "automobile", "aeroplane", "couch"}
MODIFIERS = {"old", "new", "small", "large", "big", "little", "glossy", "shiny",
             "elegant", "plunging", "beautiful", "cute", "vintage", "modern"}
COMPOUND_TAILS = {"collar", "leash", "toy", "print", "picture", "photo", "statue",
                  "logo", "costume", "shaped", "design", "pattern", "sleeve", "sleeves",
                  "art", "icon", "illustration", "drawing", "painting", "model",
                  "pot", "pots", "pottery", "planter", "planters", "bed", "beds",
                  "light", "lights", "egg", "eggs", "cage", "cages"}

# Quarantine categories with confirmed sense/caption failures until a richer
# sense resolver or human image labels are available. Do not invent a sense.
AMBIGUOUS_OBJECTS = {"iris", "plate", "vase", "ball"}
NON_PHOTO = re.compile(
    r"\b(vector|illustrations?|cartoons?|typography|drawings?|paintings?|painted|"
    r"sketch|tattoos?|templates?|posters?|icons?|logos?|cgi|animation|animated|"
    r"figurines?|statues?|toys?|miniature)\b|\b3\s*d\b|\bhand drawn\b|"
    r"\bseamless (pattern|background)\b|\bdigital (montage|art|render)\b|"
    r"\bblack and white\b|\bmonochrome\b"
)
PERSON_WORDS = set("man men woman women girl girls boy boys person people actor actress model bride baby child children female male wearing wore wears dressed".split())
FAMILIES = [
    {"flower", "rose", "tulip", "lily", "iris"},
    {"bird", "duck", "chicken", "eagle", "parrot", "penguin"},
    {"cat", "tiger", "lion"},
    {"dog", "wolf", "fox"},
    {"boat", "ship", "yacht", "canoe"},
]


def related_objects(left, right):
    return left == right or any(left in group and right in group for group in FAMILIES)


def context(caption, target):
    """Require matching explicitly named secondary objects and human presence.

    This is a caption constraint, not evidence that scene/identity is preserved.
    """
    words = normalize(caption).split()
    secondary = {noun(w) for w in words if noun(w) and not related_objects(noun(w), target)}
    if PERSON_WORDS.intersection(words):
        secondary.add("__person__")
    return frozenset(secondary)


def noun(word):
    if word in NOUNS:
        return ALIASES.get(word, word)
    if word in ALIASES and ALIASES[word] in NOUNS:
        return ALIASES[word]
    for stem in (word[:-1] if word.endswith("s") else "",
                 word[:-2] if word.endswith("es") else "",
                 word[:-3] + "y" if word.endswith("ies") else ""):
        if stem in NOUNS:
            return ALIASES.get(stem, stem)
    return None


def annotate_with_reason(caption):
    text = normalize(caption)
    if re.search(r"\b(no|not|without|neither|never)\b", text):
        return None, "negation"
    if NON_PHOTO.search(text):
        return None, "non_photo_or_monochrome_caption"
    words = text.split()
    phrases = []
    for pos, word in enumerate(words):
        obj = noun(word)
        if obj is None:
            continue
        if pos + 1 < len(words) and (words[pos + 1] in COMPOUND_TAILS or noun(words[pos + 1])):
            continue
        attrs, index, bridges = {}, pos - 1, 0
        while index >= 0:
            token = words[index]
            if token in TOKEN_KIND:
                kind = TOKEN_KIND[token]
                if kind in attrs:
                    return None, "multiple_values"  # Multi-valued attribute is ambiguous.
                attrs[kind] = ALIASES.get(token, token)
            elif token in MODIFIERS and bridges < 2:
                bridges += 1
            else:
                break
            index -= 1
        if not attrs:
            continue
        if index >= 0 and (words[index] in {"and", "or"} or words[index] in TOKEN_KIND):
            return None, "conjoined_attribute"
        if obj in AMBIGUOUS_OBJECTS:
            return None, "quarantined_object_sense"
        # e.g. red floral dog collar: a rejected compound must not become
        # a valid label through a different phrase in the same caption.
        phrases.append((obj, *(attrs.get(k, "") for k in KINDS)))
    # Background/sky/photo are not target nouns. Multiple qualified objects
    # (e.g. red dress with blue shoes) are left for a richer parser/audit.
    return (phrases[0], "accepted") if len(phrases) == 1 else (None, "no_unique_bound_attribute")


def annotate_object(caption):
    return annotate_with_reason(caption)[0]


def mine_object(records, seed, max_tuples, per_reference):
    rng = random.Random(seed)
    by_edit = defaultdict(lambda: defaultdict(list))
    # Re-validate to avoid accepting stale v3 signatures from cached callers.
    records = [dict(r, signature=annotate_object(r["caption"])) for r in records]
    records = [r for r in records if r["signature"] is not None]
    contexts = {r["image"]: context(r["caption"], r["signature"][0]) for r in records}
    mentions = {r["image"]: {noun(w) for w in normalize(r["caption"]).split() if noun(w)}
                for r in records}
    for rec in sorted(records, key=lambda r: r["image"]):
        sig = rec["signature"]
        for index in range(1, 4):
            if sig[index]:
                by_edit[(index, sig[index])][sig[0]].append(rec)
    references = sorted(records, key=lambda r: r["image"])
    rng.shuffle(references)
    rows, seen, usage = [], set(), Counter()

    def compatible(reference, candidate, edit):
        # If a preserve attribute is specified in the reference, require it
        # in the positive/CF-A caption too; absence is not proof it is preserved.
        sig = candidate["signature"]
        ref_sig = reference["signature"]
        return (contexts[reference["image"]] == contexts[candidate["image"]] and
                all(not ref_sig[j] or ref_sig[j] == sig[j] for j in range(1, 4) if j != edit))

    for round_id in range(per_reference):
        for ref in references:
            sig = ref["signature"]
            indices = [i for i in range(1, 4) if sig[i]]
            rng.shuffle(indices)
            indices.sort(key=lambda i: usage[KINDS[i - 1]])
            added = False
            for index in indices:
                # CF-A is a separate image, not a repeated reference.
                cf_as = [r for r in by_edit[(index, sig[index])][sig[0]]
                         if r["image"] != ref["image"] and compatible(ref, r, index)]
                if not cf_as:
                    continue
                afters = [v for v in VOCAB[KINDS[index - 1]] if v != "grey" and v != sig[index]]
                rng.shuffle(afters)
                for after in afters:
                    pool = by_edit.get((index, after), {})
                    positives = [r for r in pool.get(sig[0], []) if compatible(ref, r, index)
                                 and (ref["image"], r["image"], index) not in seen]
                    negatives = {obj: [r for r in pool[obj]
                        if contexts[r["image"]] == contexts[ref["image"]]
                        and not any(related_objects(sig[0], mention) for mention in mentions[r["image"]])]
                        for obj in sorted(pool) if not related_objects(obj, sig[0])}
                    negative_objects = [obj for obj, candidates in negatives.items() if candidates]
                    if not positives or not negative_objects:
                        continue
                    positive = rng.choice(positives)
                    cf_a = rng.choice(cf_as)
                    cf_b = rng.choice(negatives[rng.choice(negative_objects)])
                    kind = KINDS[index - 1]
                    preserve = {"object": sig[0], **{KINDS[j - 1]: sig[j]
                        for j in range(1, 4) if j != index and sig[j]}}
                    row = dict(zip(ROLES, [r["image"] for r in (ref, positive, cf_a, cf_b)]))
                    row.update(modification=f"change the {kind} of the {sig[0]} from {sig[index]} to {after}",
                        attribute_type=kind, before=sig[index], after=after, object=sig[0],
                        preserved_attributes=json.dumps(preserve, sort_keys=True),
                        cf_b_changed_attribute="object")
                    for role, rec in zip(ROLES, (ref, positive, cf_a, cf_b)):
                        row[f"{role}_caption"] = rec["caption"]
                    rows.append(row)
                    seen.add((ref["image"], positive["image"], index))
                    usage[kind] += 1
                    added = True
                    break
                if added:
                    break
            if len(rows) >= max_tuples:
                rng.shuffle(rows)
                return rows
        print(f"Object-type mining round {round_id + 1}/{per_reference}: {len(rows)} candidates", flush=True)
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    main(annotator=annotate_object, miner=mine_object, profile="object_type_preservation_v3_1",
         extra_limitations=[
             "CF-B changes object type, unlike strict v2 which changes a secondary attribute on the same object type.",
             "This is an object-type preservation pilot, not evidence of preserving instance identity or all other visual details.",
             "Different-object CF-B can be easier; do not compare these metrics directly with the previous dataset.",
             "Vocabulary and local phrase rules are incomplete; accepted labels still require visual audit.",
             "Iris, plate, vase and ball are quarantined due to confirmed object-sense/caption failures.",
             "Explicit non-photo/monochrome captions are excluded; unmarked illustrations/photos cannot be distinguished by these rules.",
             "Secondary caption objects and human-presence tags must match; this does not verify scene or identity preservation.",
             "Related-object and mentioned-object exclusions reduce false CF-B negatives but do not prove visual object absence.",
         ])
