import fire, hostlist, os, socket


def go():

    task_index  = int( os.environ['SLURM_PROCID'] )
    n_tasks     = int( os.environ['SLURM_NPROCS'] )
    # port        = int( os.environ['SLURM_STEP_RESV_PORTS'].split('-')[0] )

    tf_hostlist = hostlist.expand_hostlist( os.environ['SLURM_NODELIST'] )

    print('hostname: ', socket.gethostname())

    print('host list', tf_hostlist)
    print('n tasks', n_tasks)
    print('task idx ', task_index)


if __name__ == "__main__":
    fire.Fire(go)
