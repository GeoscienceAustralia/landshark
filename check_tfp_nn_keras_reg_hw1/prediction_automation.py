from optparse import OptionParser
from pathlib import Path


def generate_automation_scripts_on_48_cup_nodes(opts):
    nodes = int(opts.nodes)
    total_splits = 48 * nodes
    name = opts.name
    pred_stub = "pred_parts.sh"

    for i in range(nodes):
        start = i * 48 + 1
        end = (i + 1) * 48
        s1 = f"\nmkdir -p query_logs/"
        # s1 = f"\nparallel  mkdir -p query_{name}_strip{{1}}of{{2}} ::: {{{start}..{end}}} ::: {total_splits}"
        s2 = f"\nparallel -u -j 48 query_predict {{1}} {{2}} \">\" query_logs/${{PBS_JOBNAME}}_${{PBS_JOBID}}_{{1}}_{{2}}.log ::: {{{start}..{end}}} ::: {total_splits}"
        # s2 = f"\nparallel -u -j 48 query_predict {{1}} {{2}} \">\" query_{name}_strip{{1}}of{{2}}/${{PBS_JOBNAME}}_${{PBS_JOBID}}_{1}.log ::: {{{start}..{end}}} ::: {total_splits}"

        print(s1)
        print(s2)
        src = Path(pred_stub)
        dest = Path(str(i+1) + "_" + src.name)
        dest.write_text(src.read_text())
        with open(dest, "a") as file_object:
            file_object.write(s1)
            file_object.write(s2)


if __name__ == '__main__':

    parser = OptionParser(usage='%prog -c config_file_name \n'
                                '-h halfwidth -n name -b batchsize -e epochs -f total_folds \n'
                                '-i interations -w halfwidth')
    parser.add_option('-n', '--name', type='string', dest='name',
                      help='base name of job')
    parser.add_option('-m', '--nodes', type='string', dest='nodes',
                      help='number of nodes')
    options, args = parser.parse_args()

    generate_automation_scripts_on_48_cup_nodes(options)
