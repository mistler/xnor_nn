#!/usr/bin/env python

import sys
import os

cpu = ""
num_threads = 1

avx_cycles = {"LOAD": 1.0, "STORE": 1.0, "BCAST": 2.0, "MUL": 0.5, "ADD": 0.5, "LOGIC": 0.5}
sse3_cycles = {"LOAD": 1.0, "STORE": 1.0, "BCAST": 2.0, "MUL": 0.5, "ADD": 0.5, "LOGIC": 0.33}
neon_cycles = {"LOAD": 1.0, "STORE": 1.0, "FMA": 1.0, "BCAST": 1.0, "MUL": 1.0, "ADD": 1.0, "LOGIC": 1.0}

def div_up(a, b):
    if b == 0: raise ArithmeticError("Division by zero in div_up")
    return (a+b-1)/b

def getM(s):
    return s["OC"]

def getN(s):
    return s["OH"] * s["OW"]

def getK(s):
    return s["KH"] * s["KW"] * s["IC"]

def gemm_execute(M, N, K):
    import numpy as np
    import time

    a = np.random.rand(M, K).astype('f')
    b = np.random.rand(K, N).astype('f')

    for i in range(0, 5):
        ab = np.dot(a, b)
    iters = 100

    t1 = time.time()
    for i in range(0, iters):
        np.dot(a, b)
    t2 = time.time()
    duration = (t2-t1) / iters;
    return duration

def ops_gemm_avx(M, N, K, vlen):
    return {"MUL": M*N*K/vlen, "ADD": M*N*K/vlen, "STORE": M*N/vlen, "LOAD": M*N*K/vlen/2}

def ops_gemm_sse3(M, N, K, vlen):
    return {"MUL": M*N*K/vlen, "ADD": M*N*K/vlen, "STORE": M*N/vlen, "LOAD": M*N*K/vlen/2}

def ops_gemm_neon(M, N, K, vlen):
    return {"MUL": M*N*K/vlen, "ADD": M*N*K/vlen, "STORE": M*N/vlen, "LOAD": M*N*K/vlen/2}

def ops_conv_avx(s, vlen):
    OCO = div_up(s["OC"], vlen)
    ICO = div_up(s["IC"], 16)
    stores = OCO*s["OH"]*s["OW"]
    inner = OCO*s["OH"]*s["OW"]*s["KH"]*s["KW"]*ICO
    return {"MUL": stores*3, "ADD": stores, "STORE": stores, "BCAST": inner, "LOAD": inner, "LOGICS256": 4*inner, "LOGICS128": 31*inner+stores*16}

def ops_conv_neon(s, vlen):
    OCO = div_up(s["OC"], vlen)
    ICO = div_up(s["IC"], 16)
    stores = OCO*s["OH"]*s["OW"]
    inner = OCO*s["OH"]*s["OW"]*s["KH"]*s["KW"]*ICO
    return {"ADD": 2*inner, "STORE": stores, "BCAST": inner, "LOAD": inner, "LOGICS128": 3*inner}

def theoretical_peak_ops(mhz, vlen, num_threads, throughput, fma):
    return mhz*vlen*num_threads/throughput*(2 if fma else 1)

def i7_theoretical_peak_ops():
    return theoretical_peak_ops(2.4, 8, num_threads, 0.5, False)

def atom_theoretical_peak_ops():
    return theoretical_peak_ops(2.4, 4, num_threads, 0.5, False)

def a53_theoretical_peak_ops():
    return theoretical_peak_ops(1.2, 4, num_threads, 1, False)

def gemm_ops(M, N, K, vlen, ops_counter, duration):
    operations = ops_counter(M, N, K, vlen)
    fp = ops_to_sops(operations, vlen)
    return fp / duration / 1000 / 1000 / 1000

def conv_ops(s, vlen, ops_counter, duration):
    operations = ops_counter(s, vlen)
    fp = ops_to_sops(operations, vlen)
    return fp / duration / 1000 / 1000 / 1000

def check_add(dictionary, val, multiplier):
    return dictionary[val]*multiplier if val in dictionary else 0

def ops_to_sops(ops, vlen):
    result = 0
    result += check_add(ops, "FMA", vlen*2)
    result += check_add(ops, "MUL", vlen)
    result += check_add(ops, "ADD", vlen)
    result += check_add(ops, "LOGICS128", 4)
    result += check_add(ops, "LOGICS256", 8)
    return result

def ops_to_cycles(ops):
    result = 0
    result += check_add(ops, "FMA", c["FMA"])
    result += check_add(ops, "MUL", c["MUL"])
    result += check_add(ops, "ADD", c["ADD"])
    result += check_add(ops, "LOAD", c["LOAD"])
    result += check_add(ops, "STORE", c["STORE"])
    result += check_add(ops, "BCAST", c["BCAST"])
    result += check_add(ops, "LOGICS128", c["LOGICS128"])
    result += check_add(ops, "LOGICS256", c["LOGICS256"])
    return result

def getOH(ih, kh, sh, ph):
    return (ih + 2*ph - kh) / sh + 1

def getOW(iw, kw, sw, pw):
    return (iw + 2*pw - kw) / sw + 1

def getConvSize(line):
    line = line.replace(",", "").replace(":", "").split(" ")
    oc = int(line[6])
    ic = int(line[8])
    ih = int(line[10])
    iw = int(line[12])
    kh = int(line[14])
    kw = int(line[16])
    sh = int(line[18])
    sw = int(line[20])
    ph = int(line[22])
    pw = int(line[24])
    return {"OC": oc, "IC": ic, "OH": getOH(ih, kh, sh, ph), "OW": getOW(iw, kw, sw, pw), "KH": kh, "KW": kw}

def print_results(vlen, peak, cg, gg, conv_time, gemm_time, gemm_isa, conv_isa):
    ops_format = "{:.2f}"
    time_format = "{:7.2f}"
    speedup_format = "{:.1f}x"

    print "conv\t" + conv_isa + ":\t(" + time_format.format(conv_time*1000) + "ms) = " + ops_format.format(cg) + " Ops/s"
    print "gemm\t" + gemm_isa + ":\t(" + time_format.format(gemm_time*1000) + "ms) = " + ops_format.format(gg) + " Ops/s"
    print "current speedup:\t" + speedup_format.format(gemm_time / conv_time)
    print "potential (ct gemm):\t" + speedup_format.format(gemm_time / conv_time * gg / cg)
    print "potential (ct peak):\t" + speedup_format.format(gemm_time / conv_time * peak / cg)

def process_conv_params(conv_size, conv_time):
    M = getM(conv_size)
    N = getN(conv_size)
    K = getK(conv_size)
    gemm_time = gemm_execute(M, N, K)

    if cpu == "i7-3517u":
        vlen = 8
        i7_peak = i7_theoretical_peak_ops()
        cg = conv_ops(conv_size, vlen, ops_conv_avx, conv_time)
        gg = gemm_ops(M, N, K, vlen, ops_gemm_sse3, gemm_time)
        print_results(vlen, i7_peak, cg, gg, conv_time, gemm_time, "SSE3", "AVX")

    if cpu == "cortex-a53":
        vlen = 4
        a53_peak = a53_theoretical_peak_ops()
        cg = conv_ops(conv_size, vlen, ops_conv_neon, conv_time)
        gg = gemm_ops(M, N, K, vlen, ops_gemm_neon, gemm_time)
        print_results(vlen, a53_peak, cg, gg, conv_time, gemm_time, "NEON", "NEON")

def parse_perf():

    from subprocess import Popen, PIPE
    cmd = Popen(["./test/perf/test_simple_net"], stdout=PIPE)
    lines = cmd.stdout
    """
    with open('out.txt') as f:
        lines = f.readlines()
    """

    for line in lines:
        line = line.rstrip('\n')
        if len(line) == 0:
            print ""
            continue

        if line == "success":
            return

        if line[0].isdigit():
            print line
            conv_size = getConvSize(line)

        if line.startswith("Convolution"):
            process_conv_params(conv_size, float(line.split(" ")[3]) / 1000)

def main():
    if len(sys.argv) < 2:
        print "Please specify processor codename"
        return

    global cpu
    cpu = sys.argv[1]

    if not "OMP_NUM_THREADS" in os.environ:
        print "Please export OMP_NUM_THREADS variable"
        return
    global num_threads
    num_threads = int(os.environ["OMP_NUM_THREADS"])

    ops = 0
    if cpu == "i7-3517u":
        ops = i7_theoretical_peak_ops()
    elif cpu == "cortex-a53":
        ops = a53_theoretical_peak_ops()
    else:
        print "Please specify processor codename"
        return

    print "Theoretical peak " + str(cpu) + " (" + str(num_threads) + " threads): " + str(ops) + " Ops/s"
    parse_perf()

main()
