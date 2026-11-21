"""Independent reimplementation (no validity_screen import)."""
import math
Z = 1.959963984540054
def wilson(k, n, z=Z):
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0., c-h), min(1., c+h)
def newcombe(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    d = p1-p2
    return d - math.sqrt((p1-l1)**2 + (u2-p2)**2), d + math.sqrt((u1-p1)**2 + (p2-l2)**2)
def lchoose(n, k):
    return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
def hyp_pmf(x, N, nc, K):
    if x < max(0, K-(N-nc)) or x > min(nc, K): return 0.
    return math.exp(lchoose(nc, x) + lchoose(N-nc, K-x) - lchoose(N, K))
def midp(a, b, c, d):
    nc, N, K = a+c, a+b+c+d, a+b
    lo, hi = max(0, K-(N-nc)), min(nc, K)
    pm = [hyp_pmf(x, N, nc, K) for x in range(lo, hi+1)]
    i = a - lo
    up = sum(pm[i+1:]) + .5*pm[i]
    dn = sum(pm[:i]) + .5*pm[i]
    return up, dn
def tier(a, b, c, d):
    nc, ni = a+c, b+d
    def st(k, m):
        if m < 10: return "ne"
        if k/m >= .95: return "cert" if wilson(k, m)[0] > .90 else "susp"
        return "ok"
    sL, sF = st(b, ni), st(c, nc)
    if sL == "cert": return "Invalid", "blanket_confidence"
    if sF == "cert": return "Invalid", "blanket_withdrawal"
    if nc < 10 or ni < 10: return "Uncertifiable", "too_few_errors" if ni < 10 else "too_few_correct"
    up, dn = midp(a, b, c, d)
    if dn < .025: return "Invalid", "inversion"
    if sL == "susp": return "Indeterminate", "blanket_confidence_suspected"
    if sF == "susp": return "Indeterminate", "blanket_withdrawal_suspected"
    if up < .025: return "Valid", "signal_detected"
    if a/nc - b/ni < 0: return "Indeterminate", "inversion_suspected"
    return "Indeterminate", "no_detectable_sensitivity"
