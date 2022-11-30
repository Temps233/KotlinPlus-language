# ****************************************************************
# Copyright (c) 2022 KotlinPlus Development Team
# Copyright (c) 2019 David Callanan
# ****************************************************************

import sys, time
import Interpreter
import release
import traceback as tb
from os import environ, system, name as osname, _exit as osexit
def _exit(s: int):
    print("\n",end='',flush=True)
    osexit(s)

if len(sys.argv) == 2:
    fn = sys.argv[1]
    print(f"Loading Script '{fn}': ", end='', flush=True)
    time.sleep(0.2)
    try:
        f = open(fn, "r")
        script = f.read()
    except Exception as e:
        print(f"× Failed to load script '{fn}'")
        print(" "*(19+len(fn))+"╰─> ", end='', flush=True)
        time.sleep(0.2)
        print(str(e), end='', flush=True)
        input("\npress enter to continue...")
        _exit(0)
    else:
        print(f"Successfully loaded script {fn}\n", flush=True)
    
    input("\npress enter to run...")
    system('cls' if osname == 'nt' else 'cls')
    _, error = Interpreter.run(fn, script)
    if error:
        print(error.as_string())
    input("\npress enter to exit...")
    _exit(0)
print(f"KotlinPlus {release.release_info} on {sys.platform}")
while True:
    try:
        text = input(f'>>> ')
        RBrace = text.count("}")
        LBrace = text.count("{")
        while LBrace != RBrace:
            text += "\n"+input("... ")
            RBrace = text.count("}")
            LBrace = text.count("{")
        if text.strip() == '': continue
        if text.strip().startswith("#"): continue
        result, error = Interpreter.run(f'<Ktplus REPL>', text)
        if error:
            print(error.as_string())
        elif result:
            if len(result.elements) == 1:
                print(repr(result.elements[0]))
            else:
                print(repr(result))
    except (EOFError, KeyboardInterrupt, SystemExit) as AnyExitError:
        _exit(0)

    except BaseException as e:
        tb.print_exc()