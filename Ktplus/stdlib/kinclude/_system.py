# ****************************************************************
# Copyright (c) 2022 KotlinPlus Development Team
# ****************************************************************
import stdlib.KtplusAPI as KtplusAPI
import os
def main(self, exec_ctx):
    command = exec_ctx.symbol_table.get('command')
    result = os.system(str(command))
    return KtplusAPI.RTResult().success(KtplusAPI.Number(result))

main.arg_names = ['command']