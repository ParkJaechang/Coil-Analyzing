import pathlib, sys
path = pathlib.Path(r"D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\pyrasite_test.txt")
path.write_text("ok\n" + "\n".join(sorted(list(sys.modules)[:50])), encoding="utf-8")
