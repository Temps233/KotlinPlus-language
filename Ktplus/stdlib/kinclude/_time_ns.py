# ****************************************************************
# Copyright (c) 2022 KotlinPlus Development Team
# ****************************************************************
import stdlib.KtplusAPI as KtplusAPI
import time
def main(self, exec_ctx):
    return KtplusAPI.RTResult().success(KtplusAPI.Number(time.perf_counter_ns()))
main.arg_names = []