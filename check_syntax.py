import py_compile
try:
    py_compile.compile('app.py', doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"Syntax Error: {e}")
except Exception as e:
    print(f"Error: {e}")
