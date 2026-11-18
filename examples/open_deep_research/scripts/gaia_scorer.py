import re
import string
import warnings


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element: any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    # if gt is a number
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)


def check_prediction_contains_answer_letters_in_order(prediction, true_answer):
    prediction = prediction.lower()
    true_answer = true_answer.lower()
    if len(prediction) > len(true_answer) * 3:
        return False
    i = 0
    for letter in true_answer:
        if letter in prediction[i:]:
            i += prediction[i:].index(letter)
        else:
            return False
    return True


def check_close_call(prediction, true_answer, is_correct):
    if is_correct:
        return True
    else:
        if is_float(true_answer):
            return is_correct
        else:
            if (
                check_prediction_contains_answer_letters_in_order(str(prediction), str(true_answer))
                and len(str(true_answer)) * 0.5 <= len(str(prediction)) <= len(str(true_answer)) * 2
            ):
                print(f"Close call: {prediction} vs {true_answer}")
                return True
            else:
                return False


_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def normalize_answer_text(s: str) -> str:
    """Unicode NFKC, every dash -> '-', curly quotes -> straight. Formatting only, the answer itself is not changed."""
    import unicodedata
    return unicodedata.normalize("NFKC", str(s)).translate(_DASHES).translate(_QUOTES).strip()


def _scaled_number(s: str):
    """'100 million' -> 100000000.0, '1.5 billion' -> 1500000000.0; None if not '<number> <scale word>'."""
    m = re.fullmatch(r"\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(thousand|million|billion|trillion)\s*", s.lower())
    return float(m.group(1).replace(",", "")) * _SCALES[m.group(2)] if m else None


# Accepted alternative answers, reviewed by hand (ground truth -> other names of the same thing). Keep this list short and
# documented: each entry is a judgment call, see notes/2026-11-18_go3v3_cheap20_scoring_observations.md.
ANSWER_ALIASES = {
    "Russian-German Legion": ["Russo-German Legion"],   # GAIA 2023 validation, Whitney 2022.128 question: same unit, both English names
}
# Per-question accepted answers (task_id -> alternatives), for ground truths that are stale or incomplete. Hand-reviewed.
TASK_ALIASES = {
    # Box Office Mojo 2020 worldwide vs domestic top 10: the live lists (checked 2026-11-18) share 3 titles (Bad Boys for Life,
    # Sonic the Hedgehog, Dolittle); the ground truth 6 reflects the lists when the question was written. Date-dependent.
    "2dfc4c37-fec1-4518-84a7-10095d30ad75": ["3"],
    # Pearl City homes: the question asks which home sold for more AND the price; naming the home with the price answers it fully.
    "f2feb6a4-363c-4c09-a804-0db564eafd68": ["2017 Komo Mai Drive 900000"],
}


def relaxed_question_scorer(model_answer, ground_truth, task_id=None) -> bool:
    """Official GAIA question_scorer after formatting normalization (dashes, quotes, numbers with a scale word), also
    accepting the hand-reviewed ANSWER_ALIASES. The whole answer is scored: text after the answer still makes it wrong.
    Never raises, prints or warns."""
    gt = normalize_answer_text(ground_truth)
    alts = [gt] + ANSWER_ALIASES.get(gt, []) + TASK_ALIASES.get(task_id, [])
    return any(_relaxed_one(model_answer, alt) for alt in alts)


def _relaxed_one(model_answer, ground_truth) -> bool:
    import contextlib, io
    pred, gt = normalize_answer_text(model_answer or ""), normalize_answer_text(ground_truth)
    if is_float(gt) and _scaled_number(pred) is not None:
        return _scaled_number(pred) == float(gt)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(question_scorer(pred, gt))
        except Exception:
            return False


def first_line(prediction) -> str:
    """Text after a 'FINAL ANSWER:' marker if present, else the first non-empty line (comparison column only)."""
    p = str(prediction or "")
    m = re.search(r"final answer\s*:\s*(.+)", p, re.I)
    if m:
        return m.group(1).strip().splitlines()[0]
    return next((l.strip() for l in p.splitlines() if l.strip()), "")


def rescore(result: dict) -> dict:
    """Add the three scores to a per-question result; `is_correct` becomes the relaxed score (exact match kept as is_correct_exact)."""
    pred, gt, tid = result.get("prediction"), str(result.get("true_answer", "")), result.get("task_id")
    exact = result.get("is_correct_exact", result.get("is_correct", False))
    return {**result, "is_correct_exact": bool(exact), "is_correct": relaxed_question_scorer(pred, gt, tid),
            "is_correct_first_line": relaxed_question_scorer(first_line(pred), gt, tid), "scorer": "relaxed_v3"}


def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()
