from pathlib import Path
from subprocess import run
nodes = 10
total_splits = 48 * nodes
name = "sirsam"
pred_stub = "pred_parts.sh"

for i in range(nodes):
    start = i * 48 + 1
    end = (i + 1) * 48
    s1 = f"\nparallel  mkdir -p query_${name}_strip{{1}}of{{2}} ::: {{{start}..{end}}} ::: {total_splits}"
    s2 = f"\nparallel -u -j 48 query_predict {{1}} {{2}} \">\" query_${name}_strip{1}of{2}/${{PBS_JOBNAME}}_${{PBS_JOBID}}_{1}.log ::: {{{start}..{end}}} ::: {total_splits}"

    print(s1)
    print(s2)
    src = Path(pred_stub)
    dest = Path(str(i) + "_" + src.name)
    dest.write_text(src.read_text())
    with open(dest, "a") as file_object:
        file_object.write(s1)
        file_object.write(s2)
        run(f"qsub {dest.as_posix()}", shell=True)
