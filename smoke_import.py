import traceback

try:
    import numbers3_logic as nl
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    print('IMPORT_FAILED')
    raise
