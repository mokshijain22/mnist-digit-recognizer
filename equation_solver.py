"""
equation_solver.py  (v6)
------------------------
Fixes over v5:
  1. _prep_for_cnn: aspect-ratio preserving resize (was squashing '1' → '0').
     tight-crop → 4px pad → scale longest side to 20px → centre in 28×28.
  2. build_expression: strip leading zeros from each number token so
     Python AST does not raise "leading zeros not permitted" on e.g. "040".
  3. _plus_score: tightened Gate 1/2 thresholds so tall '+' strokes
     (drawn large on canvas) are not rejected by the narrow-stripe gates.
  4. _score_operators: no aspect early-exit (was blocking extreme ratios).
     Each scorer handles its own geometry. Minus scorer relaxed to aspect>1.5.
"""

import cv2
import numpy as np
from PIL import Image
import ast
import operator as op_module
import re
import logging

print("[equation_solver] v6 loaded OK")

# ---------------------------------------------------------------------------
# Safe math evaluator
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add:      op_module.add,
    ast.Sub:      op_module.sub,
    ast.Mult:     op_module.mul,
    ast.Div:      op_module.truediv,
    ast.Pow:      op_module.pow,
    ast.USub:     op_module.neg,
    ast.UAdd:     op_module.pos,
    ast.FloorDiv: op_module.floordiv,
    ast.Mod:      op_module.mod,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        fn = _OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported op: {node.op}")
        l, r = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Div) and r == 0:
            raise ZeroDivisionError("Division by zero")
        return fn(l, r)
    if isinstance(node, ast.UnaryOp):
        fn = _OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported unary: {node.op}")
        return fn(_eval_node(node.operand))
    raise ValueError(f"Unsupported node: {type(node)}")


def safe_evaluate(expr: str):
    """Returns (result, error_string). result is None on error."""
    try:
        result = _eval_node(ast.parse(expr, mode='eval').body)
        if isinstance(result, float):
            result = round(result, 10)
            if result == int(result):
                result = int(result)
        return result, None
    except ZeroDivisionError:
        return None, "Division by zero"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Expression builder + validator
# ---------------------------------------------------------------------------

_VALID_OPERATORS = set('+-*/')


def _strip_leading_zeros(expr: str) -> str:
    """
    Strip leading zeros from every integer literal in the expression.
    e.g.  "040+01"  →  "40+1"
    Handles multi-digit runs: "0001" → "1", "00" → "0".
    Does NOT touch numbers after a decimal point (no floats here but safe).
    """
    def _fix(m):
        s = m.group(0).lstrip('0') or '0'
        return s
    return re.sub(r'\b0+(\d+)', lambda m: m.group(0).lstrip('0') or '0', expr)


def build_expression(tokens: list) -> tuple:
    """
    Convert raw token list → clean validated expression string.
    Returns (expression_str, error_str). error_str is None on success.
    """
    raw_symbols = []
    for t in tokens:
        if not isinstance(t, dict):
            continue
        sym = str(t.get('symbol', '')).strip()
        if not sym or sym == '=':
            continue
        raw_symbols.append(sym)

    if not raw_symbols:
        return '', 'No valid symbols detected'

    if not any(s.isdigit() for s in raw_symbols):
        return '', 'No digits detected — only operators found'

    joined  = ''.join(raw_symbols)
    cleaned = re.sub(r'[^0-9+\-*/]', '', joined)

    if not cleaned:
        return '', 'Expression contains no recognisable digits or operators'

    # Collapse consecutive operators → keep last one
    prev = None
    while prev != cleaned:
        prev    = cleaned
        cleaned = re.sub(r'[+\-*/]{2,}', lambda m: m.group(0)[-1], cleaned)

    cleaned = cleaned.strip('+-*/')
    while cleaned and cleaned[0] in _VALID_OPERATORS:
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] in _VALID_OPERATORS:
        cleaned = cleaned[:-1]

    if not cleaned or not any(c.isdigit() for c in cleaned):
        return '', 'Expression has no digits after cleaning'

    # FIX: strip leading zeros so AST doesn't raise "leading zeros not permitted"
    # e.g. "040+1" → "40+1",  "0+0" stays "0+0"
    cleaned = _strip_leading_zeros(cleaned)

    print(f'[build_expression] raw_symbols : {raw_symbols}')
    print(f'[build_expression] expression  : {cleaned}')
    logging.debug('[build_expression] expression: %s', cleaned)

    return cleaned, None


# ---------------------------------------------------------------------------
# Preprocessing — minimal, Otsu only
# ---------------------------------------------------------------------------

def _to_binary(image) -> np.ndarray:
    """
    White-on-black binary image via Otsu threshold.
    NO dilation — even a 2×2 kernel merges close symbols and distorts shapes.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert('L'))
    elif isinstance(image, np.ndarray):
        arr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    else:
        raise TypeError(f"Unsupported type: {type(image)}")

    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    print(f'[_to_binary] shape={arr.shape} ink_px={np.sum(binary>0)}')
    return binary


# ---------------------------------------------------------------------------
# Valley splitter
# ---------------------------------------------------------------------------

def _split_at_valleys(binary: np.ndarray, box, min_valley_width: int = 4):
    x, y, w, h = box
    roi      = binary[y:y+h, x:x+w]
    col_proj = np.sum(roi > 0, axis=0).astype(float)
    if w > 5:
        col_proj = np.convolve(col_proj, np.ones(3)/3, mode='same')

    peak        = col_proj.max() or 1
    valley_mask = col_proj <= peak * 0.05

    valleys = []
    in_v, v_start = False, 0
    for i, is_v in enumerate(valley_mask):
        if is_v and not in_v:
            in_v, v_start = True, i
        elif not is_v and in_v:
            in_v = False
            if i - v_start >= min_valley_width:
                valleys.append((v_start + i) // 2)
    if in_v and w - v_start >= min_valley_width:
        valleys.append((v_start + w) // 2)

    if not valleys:
        return [box]

    splits    = [0] + valleys + [w]
    sub_boxes = []
    for i in range(len(splits)-1):
        sx, ex = splits[i], splits[i+1]
        sw     = ex - sx
        if sw < 5:
            continue
        slc  = roi[:, sx:ex]
        rows = np.any(slc > 0, axis=1)
        if not np.any(rows):
            continue
        ry_min = int(np.argmax(rows))
        ry_max = int(len(rows)-1-np.argmax(rows[::-1]))
        sub_boxes.append((x+sx, y+ry_min, sw, ry_max-ry_min+1))
    return sub_boxes if sub_boxes else [box]


# ---------------------------------------------------------------------------
# Bounding-box extraction
# ---------------------------------------------------------------------------

def _merge_dot_fragments(boxes, size_ratio=0.08, gap=2):
    """
    Only absorb a box into its right neighbour when it is a tiny speck:
      - gap ≤ 2px  AND  left-box area < 8% of right-box area.
    Much less aggressive than earlier versions to prevent '1' being eaten.
    """
    if len(boxes) < 2:
        return boxes
    areas  = [b[2]*b[3] for b in boxes]
    result = []
    i = 0
    while i < len(boxes):
        if i+1 < len(boxes):
            x,  y,  w,  h  = boxes[i]
            nx, ny, nw, nh = boxes[i+1]
            gap_px = nx - (x+w)
            if gap_px <= gap and areas[i] < size_ratio * areas[i+1]:
                mx = min(x, nx);  my = min(y, ny)
                mr = max(x+w, nx+nw); mb = max(y+h, ny+nh)
                boxes[i+1] = (mx, my, mr-mx, mb-my)
                areas[i+1] = (mr-mx)*(mb-my)
                i += 1
                continue
        result.append(boxes[i])
        i += 1
    return result


def get_bounding_boxes(binary: np.ndarray, min_area: int = 20):
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            boxes.append((x, y, w, h))

    if not boxes:
        print('[get_bounding_boxes] no boxes found')
        return []

    boxes.sort(key=lambda b: b[0])
    print(f'[get_bounding_boxes] raw ({len(boxes)}): {boxes}')

    boxes = _merge_dot_fragments(boxes)

    if len(boxes) >= 2:
        widths   = sorted([b[2] for b in boxes])
        median_w = widths[len(widths)//2]
        expanded = []
        for b in boxes:
            if b[2] > 2.5 * median_w:
                splits = _split_at_valleys(binary, b)
                print(f'[get_bounding_boxes] split {b} → {splits}')
                expanded.extend(splits)
            else:
                expanded.append(b)
        expanded.sort(key=lambda b: b[0])
        boxes = expanded

    print(f'[get_bounding_boxes] final ({len(boxes)}): {boxes}')
    return boxes


# ---------------------------------------------------------------------------
# Operator scoring
# ---------------------------------------------------------------------------

def _plus_score(roi: np.ndarray) -> float:
    """
    Score [0,1] for how '+'-like this ROI is.

    A large hand-drawn '+' on a 400×300 canvas will be much bigger than
    an MNIST-style symbol. The stripe gates now use adaptive thresholds
    based on the actual ROI size rather than fixed fractions.
    """
    h, w = roi.shape
    if h < 8 or w < 8:
        return 0.0

    density = np.sum(roi > 0) / (h * w)
    if density > 0.65:
        return 0.0

    row_proj = np.sum(roi > 0, axis=1).astype(float)
    col_proj = np.sum(roi > 0, axis=0).astype(float)
    row_n    = row_proj / (row_proj.max() or 1)
    col_n    = col_proj / (col_proj.max() or 1)

    # Gate 1: vertical bar — few columns have ink
    # For large '+' on canvas the bar can be ~15% of width, so allow up to 0.45
    ink_cols_frac = float(np.sum(col_n > 0.25)) / w
    if ink_cols_frac > 0.45:
        return 0.0

    # Gate 2: horizontal bar — few rows have ink
    ink_rows_frac = float(np.sum(row_n > 0.25)) / h
    if ink_rows_frac > 0.45:
        return 0.0

    # Gate 3: both bars cross the centre of the ROI
    cy_lo, cy_hi = h // 4, 3 * h // 4
    cx_lo, cx_hi = w // 4, 3 * w // 4
    if row_n[cy_lo:cy_hi].max() < 0.55:
        return 0.0
    if col_n[cx_lo:cx_hi].max() < 0.50:
        return 0.0

    # Gate 4: ink in all four quadrants ('+' has ink everywhere, '4' does not)
    q = [
        float(np.sum(roi[:h//2, :w//2] > 0)),
        float(np.sum(roi[:h//2, w//2:] > 0)),
        float(np.sum(roi[h//2:, :w//2] > 0)),
        float(np.sum(roi[h//2:, w//2:] > 0)),
    ]
    q_total     = sum(q) or 1
    min_q_share = min(q) / (q_total / 4)
    if min_q_share < 0.35:
        return 0.0

    score  = 0.55
    score += max(0.0, 0.20 * (0.45 - ink_cols_frac) / 0.45)
    score += max(0.0, 0.15 * (0.45 - ink_rows_frac) / 0.45)
    score += min(0.10, (min_q_share - 0.35) * 0.20)
    return min(score, 1.0)


def _score_operators(roi: np.ndarray) -> dict:
    """
    Score every operator. Always returns a dict — never a float.
    No aspect early-exit: each scorer applies its own geometry gates.
    """
    h, w = roi.shape
    if h < 5 or w < 5:
        return {op: 0.0 for op in '+-*/='}

    aspect  = w / max(h, 1)
    density = np.sum(roi > 0) / max(h * w, 1)

    row_proj = np.sum(roi > 0, axis=1).astype(float)
    col_proj = np.sum(roi > 0, axis=0).astype(float)
    row_n    = row_proj / (row_proj.max() or 1)
    col_n    = col_proj / (col_proj.max() or 1)

    scores = {}

    # ── Minus ('-') ──────────────────────────────────────────────────
    # Wide + flat: aspect > 1.5, thin rows, spans most columns.
    # No upper aspect limit — real minus strokes can be very wide.
    minus_s       = 0.0
    ink_rows_frac = float(np.sum(row_n > 0.30)) / max(h, 1)
    ink_cols_frac = float(np.sum(col_n > 0.30)) / max(w, 1)
    if aspect > 1.5 and ink_rows_frac < 0.55 and ink_cols_frac > 0.40:
        minus_s += min(0.55, (aspect - 1.5) / 4.0 * 0.55)
        minus_s += max(0.0, 0.30 * (1.0 - ink_rows_frac / 0.55))
        minus_s += max(0.0, 0.15 * (ink_cols_frac - 0.40) / 0.60)
    scores['-'] = min(minus_s, 1.0)

    # ── Equals ('=') ────────────────────────────────────────────────
    eq_s = 0.0
    if h > 14:
        smooth = np.convolve(row_n, np.ones(5)/5, mode='same')
        above  = smooth > 0.30
        rises  = [i for i in range(1, len(above)) if above[i]   and not above[i-1]]
        falls  = [i for i in range(1, len(above)) if above[i-1] and not above[i]]
        if len(rises) >= 2 and len(falls) >= 2:
            gap_rows = smooth[falls[0]:rises[1]]
            if len(gap_rows) > 0 and gap_rows.max() < 0.25:
                eq_s  = 0.55
                eq_s += min(0.25, (1.0 - gap_rows.max()) * 0.25)
                eq_s += min(0.20, (aspect - 0.8) / 2.0 * 0.20)
    scores['='] = min(eq_s, 1.0)

    # ── Plus ('+') ──────────────────────────────────────────────────
    scores['+'] = _plus_score(roi)

    # ── Multiply ('*') ──────────────────────────────────────────────
    mul_s = 0.0
    if 0.5 < aspect < 2.0 and density > 0.06:
        centre   = roi[h//4:3*h//4, w//4:3*w//4]
        centre_d = np.sum(centre > 0) / max(centre.size, 1)
        if centre_d < 0.40:
            d1 = sum(roi[min(h-1,int(i*h/7)), min(w-1,int(i*w/7))]     > 0 for i in range(8))
            d2 = sum(roi[min(h-1,int(i*h/7)), min(w-1,int((7-i)*w/7))] > 0 for i in range(8))
            mul_s = (d1+d2)/16.0 * (1.0-centre_d)
    scores['*'] = min(mul_s, 1.0)

    # ── Division ('/') ──────────────────────────────────────────────
    div_s = 0.0
    if aspect < 0.70 and density > 0.05:
        ink_cols_div = float(np.sum(col_n > 0.30)) / max(w, 1)
        if ink_cols_div >= 0.40:   # diagonal spreads across many columns
            div_s += max(0.0, 0.60 * (1.0 - aspect / 0.70))
            div_s += min(0.40, density * 2.0)
    scores['/'] = min(div_s, 1.0)

    return scores


# ---------------------------------------------------------------------------
# ROI → 28×28  (MNIST-style, aspect-ratio preserving)
# ---------------------------------------------------------------------------

def _prep_for_cnn(roi: np.ndarray) -> np.ndarray:
    """
    Tight-crop the ink → add 4px border → scale longest side to 20px
    (preserving aspect ratio) → centre in 28×28 → normalise [0,1].

    This is the critical fix for '1' being read as '0':
    The OLD code did cv2.resize(roi, (28,28)) which SQUASHED a tall thin
    '1' into a square blob that looks like '0' to the CNN.
    The NEW code scales proportionally so '1' stays thin.
    """
    # Tight crop to ink bounding box
    rows = np.any(roi > 0, axis=1)
    cols = np.any(roi > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    r_min = int(np.argmax(rows))
    r_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c_min = int(np.argmax(cols))
    c_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
    cropped = roi[r_min:r_max+1, c_min:c_max+1]

    # Add uniform padding so digit doesn't touch the edges
    padded = cv2.copyMakeBorder(cropped, 4, 4, 4, 4,
                                cv2.BORDER_CONSTANT, value=0)

    # Scale longest side to 20px, preserve aspect ratio
    ph, pw = padded.shape
    scale  = 20.0 / max(ph, pw)
    new_h  = max(1, int(round(ph * scale)))
    new_w  = max(1, int(round(pw * scale)))
    scaled = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centre in 28×28 canvas
    canvas       = np.zeros((28, 28), dtype=np.float32)
    y_off        = (28 - new_h) // 2
    x_off        = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = scaled.astype(np.float32)

    canvas /= 255.0
    return canvas.reshape(1, 28, 28, 1)


# ---------------------------------------------------------------------------
# Symbol classification
# ---------------------------------------------------------------------------

_OP_CERTAIN  = 0.72
_CNN_CERTAIN = 0.85


def _classify_symbol(roi: np.ndarray, model) -> dict:
    """
    Run CNN + all operator heuristics, pick winner by confidence face-off.
    Decision matrix:
      op  >= _OP_CERTAIN  and cnn < _CNN_CERTAIN  → operator
      cnn >= _CNN_CERTAIN and op  < _OP_CERTAIN   → digit
      both certain or both uncertain               → higher score wins
    """
    h, w = roi.shape
    print(f'[classify] ROI {w}×{h}  aspect={w/max(h,1):.2f}')

    prob     = model.predict(_prep_for_cnn(roi), verbose=0)[0]
    digit    = int(np.argmax(prob))
    cnn_conf = float(prob[digit])
    print(f'[classify] CNN digit={digit}  conf={cnn_conf:.3f}')

    op_scores     = _score_operators(roi)
    best_op       = max(op_scores, key=op_scores.get)
    best_op_score = op_scores[best_op]
    scores_str    = '  '.join(f'{k}:{v:.2f}' for k, v in op_scores.items())
    print(f'[classify] ops best={best_op!r}({best_op_score:.3f})  [{scores_str}]')

    op_certain  = best_op_score >= _OP_CERTAIN
    cnn_certain = cnn_conf      >= _CNN_CERTAIN

    if op_certain and not cnn_certain:
        decision, reason = 'operator', 'heuristic_certain'
    elif cnn_certain and not op_certain:
        decision, reason = 'digit', 'cnn_certain'
    elif best_op_score >= cnn_conf:
        decision = 'operator'
        reason   = 'score_higher' if best_op_score > cnn_conf else 'tie_op'
    else:
        decision, reason = 'digit', 'score_higher'

    print(f'[classify] → {decision} ({reason})')

    if decision == 'operator':
        return {'symbol': best_op, 'type': 'operator',
                'confidence': round(best_op_score, 4),
                'cnn_digit': digit, 'cnn_conf': round(cnn_conf, 4)}
    return {'symbol': str(digit), 'type': 'digit',
            'confidence': round(cnn_conf, 4),
            'op_scores': {k: round(v, 4) for k, v in op_scores.items()}}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_equation_image(image, model, min_area: int = 20) -> dict:
    """
    Full pipeline: image → binary → segment → classify → evaluate.
    """
    print('=' * 60)
    print('[solve_equation_image] pipeline start')

    binary = _to_binary(image)
    boxes  = get_bounding_boxes(binary, min_area=min_area)

    if not boxes:
        return {'tokens': [], 'expression': '', 'result': None,
                'error': 'No symbols detected — canvas appears empty',
                'boxes': []}

    tokens = []
    for idx, (x, y, w, h) in enumerate(boxes):
        roi = binary[y:y+h, x:x+w]
        print(f'\n[solve] symbol {idx}  box=({x},{y},{w},{h})')
        sym         = _classify_symbol(roi, model)
        sym['bbox'] = (x, y, w, h)
        tokens.append(sym)
        print(f'[solve] → "{sym["symbol"]}"  {sym["type"]}  conf={sym["confidence"]:.3f}')

    print(f'\n[solve] tokens: {[t["symbol"] for t in tokens]}')

    expression, build_error = build_expression(tokens)

    if build_error:
        return {'tokens': tokens, 'expression': '', 'result': None,
                'error': build_error,
                'boxes': [t['bbox'] for t in tokens]}

    result, eval_error = safe_evaluate(expression)
    print(f'[solve] expression={expression!r}  result={result}  err={eval_error}')
    print('=' * 60)

    return {'tokens': tokens, 'expression': expression,
            'result': result, 'error': eval_error,
            'boxes': [t['bbox'] for t in tokens]}