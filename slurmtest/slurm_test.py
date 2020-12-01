import fire, hostlist, os, socket


def go():

    task_index  = int( os.environ['SLURM_PROCID'] )
    n_tasks     = int( os.environ['SLURM_NPROCS'] )
    port        = int( os.environ['SLURM_STEP_RESV_PORTS'].split('-')[0] )

    tf_hostlist = [ ("%s:%s" % (host,port)) for host in hostlist.expand_hostlist( os.environ['SLURM_NODELIST']) ]

    print(socket.gethostname())
    print(tf_hostlist)

if __name__ == "__main__":
    fire.Fire(go)
