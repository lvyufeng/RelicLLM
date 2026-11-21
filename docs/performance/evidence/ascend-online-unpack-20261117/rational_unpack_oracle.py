"""Close the loop: emulate unpack.py's actual table construction (signed floats,
math.copysign) and compare against the exact oracle and the kernel body."""
from fractions import Fraction as F
import math, struct

def e4m3_value(byte):
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        magnitude = float("nan")
    elif exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return math.copysign(magnitude, -1.0 if byte & 128 else 1.0)

FP8_VALUES = tuple(e4m3_value(b) for b in range(256))
E8M0_VALUES = tuple(math.ldexp(1.0, b - 127) for b in range(255)) + (float("nan"),)

def bf16_storage_bits(value):
    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        packed = struct.pack("<f", math.copysign(float("inf"), value))
    bits = struct.unpack("<I", packed)[0]
    if bits & 0x7F800000 == 0x7F800000:
        encoded = (bits >> 16) | (0x40 if bits & 0x7FFFFF else 0)
    else:
        encoded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return encoded if encoded < 0x8000 else encoded - 0x10000

JOINT = tuple(bf16_storage_bits(code * scale) for code in FP8_VALUES for scale in E8M0_VALUES)

def where(c, x, y): return x if c else y
def mxm(a, b): return a if a > b else b
def mnm(a, b): return a if a < b else b
def kernel_bits(code, scale_code):
    sign = (code & 128) << 8
    exponent = (code >> 3) & 15
    mantissa = code & 7
    significand = where(exponent == 0, mantissa, 8 + mantissa)
    power = where(exponent == 0, -9, exponent - 10) + scale_code - 127
    msb = where(significand >= 8, 3, where(significand >= 4, 2, where(significand >= 2, 1, 0)))
    biased = power + msb + 127
    normal = (mnm(mxm(biased, 0), 255) << 7) | ((significand << (7 - msb)) & 127)
    shift = power + 133
    left = mnm(mxm(shift, 0), 7); right = mnm(mxm(-shift, 1), 4)
    truncated = significand >> right
    remainder = significand & ((1 << right) - 1)
    halfway = 1 << (right - 1)
    increment = (remainder > halfway) or ((remainder == halfway) and ((truncated & 1) != 0))
    rounded = truncated + (1 if increment else 0)
    subnormal = where(shift >= 0, significand << left, rounded)
    magnitude = where(biased <= 0, subnormal, where(biased >= 255, 0x7F80, normal))
    magnitude = where(significand == 0, 0, magnitude)
    is_nan = ((code & 127) == 127) or (scale_code == 255)
    magnitude = where(is_nan, 0x7FC0, magnitude)
    return (sign | magnitude) & 0xFFFF

# exact oracle (from oracle2)
def e4m3_exact(byte):
    sign = -1 if byte & 128 else 1
    e, m = (byte >> 3) & 15, byte & 7
    if e == 15 and m == 7: return None, sign
    mag = F(m, 2**9) if e == 0 else F(8 + m, 8) * F(2) ** (e - 7)
    return F(sign) * mag, sign
def round_rne(r):
    k = r.numerator // r.denominator; frac = r - k
    return k + 1 if frac > F(1, 2) else (k if frac < F(1, 2) else (k if k % 2 == 0 else k + 1))
def bf16_exact(m, neg):
    sign = 0x8000 if neg else 0
    if m == 0: return sign
    p = m.numerator.bit_length() - m.denominator.bit_length()
    if F(2) ** p > m: p -= 1
    if p >= -126:
        q = F(2) ** (p - 7); k = round_rne(m / q)
        if k == 256:
            p += 1
            return sign | 0x7F80 if p > 127 else sign | ((p + 127) << 7)
        if p > 127:
            return sign | 0x7F80
        return sign | ((p + 127) << 7) | (k - 128)
    k = round_rne(m / F(2) ** -133)
    return sign | (1 << 7) if k >= 128 else sign | k

table_bad, kernel_bad, disagree, nan_bad = [], [], [], []
for code in range(256):
    v, vsign = e4m3_exact(code)
    for scale in range(256):
        s = None if scale == 255 else F(2) ** (scale - 127)
        idx = code * 256 + scale
        table = JOINT[idx] & 0xFFFF
        got = kernel_bits(code, scale)
        if v is None or s is None:
            for name, val in (("table", table), ("kernel", got)):
                if not ((val & 0x7F80) == 0x7F80 and val & 0x7F):
                    nan_bad.append((code, scale, name, hex(val)))
            continue
        prod = v * s
        neg = (prod < 0) or (prod == 0 and vsign < 0)
        want = bf16_exact(abs(prod), neg)
        if table != want: table_bad.append((code, scale, hex(table), hex(want)))
        if got != want: kernel_bad.append((code, scale, hex(got), hex(want)))
        if (table & 0x7FFF) != (got & 0x7FFF):
            disagree.append((code, scale, hex(table), hex(got)))
print("joint table vs exact IEEE RNE :", len(table_bad), table_bad[:4])
print("kernel vs exact IEEE RNE      :", len(kernel_bad), kernel_bad[:4])
print("table vs kernel magnitude     :", len(disagree), disagree[:4])
print("NaN classification failures   :", len(nan_bad), nan_bad[:4])
print("signed zero table/kernel      :", hex(JOINT[0*256+127] & 0xFFFF), hex(JOINT[128*256+127] & 0xFFFF),
      hex(kernel_bits(0x00, 127)), hex(kernel_bits(0x80, 127)))

# FP4 half-scale table vs exact, and kernel expression vs exact (FP32 bits)
HALF = tuple(math.ldexp(1.0, b - 128) for b in range(255)) + (float("nan"),)
def fp32_bits(v):
    try: bits = struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError: bits = 0xFF800000 if v < 0 else 0x7F800000
    return bits
def kernel_scale_bits(code):
    bits = where(code == 0, 0x00200000, where(code == 1, 0x00400000, (mxm(code, 1) - 1) << 23))
    bits = where(code == 255, 0x7FC00000, bits)
    return bits & 0xFFFFFFFF
sfail = []
for b in range(255):
    want = fp32_bits(F(2) ** (b - 128))
    if fp32_bits(HALF[b]) != want: sfail.append(("table", b, hex(fp32_bits(HALF[b])), hex(want)))
    if kernel_scale_bits(b) != want: sfail.append(("kernel", b, hex(kernel_scale_bits(b)), hex(want)))
print("fp4 half-scale (table+kernel) vs exact FP32 2**(b-128):", len(sfail), sfail[:4])

# FP4 exact widening through the kernel helper
def kernel_e2m1(code):
    magnitude = code & 7
    high = (2 + (magnitude & 1)) << mxm((magnitude >> 1) - 1, 0)
    absolute = where(magnitude < 4, magnitude, high)
    return where((code & 8) != 0, -absolute, absolute)
bad = [c for c in range(16) if kernel_e2m1(c) != int(F(2) * (F(2) ** -1) * 0 + 0) and False]
want = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
print("fp4 exact x2 widening         :", "PASS" if [kernel_e2m1(c) for c in range(16)] == want else "FAIL")
