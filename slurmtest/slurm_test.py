import fire, hostlist, os, socket


def go():

    task_index  = int( os.environ['SLURM_PROCID'] )
    n_tasks     = int( os.environ['SLURM_NPROCS'] )
    # port        = int( os.environ['SLURM_STEP_RESV_PORTS'].split('-')[0] )

    tf_hostlist = hostlist.expand_hostlist( os.environ['SLURM_NODELIST'] )

    with open(f'log-{task_index}', 'w') as file:
        print('hostname: ', socket.gethostname(), file=file)

        print('host list', tf_hostlist, file=file)
        print('n tasks', n_tasks, file=file)
        print('task idx ', task_index, file=file)

if __name__ == "__main__":
    fire.Fire(go)
