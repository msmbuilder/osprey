Batch Submission
================

Multiple ``osprey worker`` processes can be run simultaneously and connect
to the same trials database. The following scripts might be useful as templates
for submitting multiple parallel ``osprey worker`` s to a cluster batch scheduling
system. Depending on what scheduling software your cluster runs, you can use these
scripts as a jumping off point.


Example PBS/TORQUE Script
-------------------------

.. code-block:: bash

    #!/bin/bash
    #PBS -S /bin/bash
    #PBS -l nodes=1:ppn=16
    #PBS -l walltime=12:00:00
    #PBS -V

    cd $PBS_O_WORKDIR
    NO_OF_CORES=`cat $PBS_NODEFILE | egrep -v '^#'\|'^$' | wc -l | awk '{print $1}'`
    for i in `seq $NO_OF_CORES`; do
        osprey worker config.yaml -n 100 > osprey.$PBS_JOBID.$i.log 2>&1 &
    done
    wait

Example SGE Script
------------------

.. code-block:: bash

    #!/bin/bash
    #
    #$ -cwd
    #$ -j y
    #$ -S /bin/bash
    #$ -t 1-10

    # handle if we are or are not part of an array job
    if [ "$SGE_TASK_ID" = "undefined" ]; then
        SGE_TASK_ID=0
    fi

    osprey worker config.yaml -n 100 > osprey.$JOB_ID.$SGE_TASK_ID.log 2>&1


Example SLURM Script
--------------------

.. code-block:: bash

    #!/bin/bash
    #SBATCH --time=12:00:00
    #SBATCH --mem=4000
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=16

    NO_OF_CORES=$(expr $SLURM_TASKS_PER_NODE \* $SLURM_JOB_NUM_NODES)

    for i in `seq $NO_OF_CORES`; do
        srun -n 1 osprey worker config.yaml -n 100 > osprey.$SLURM_JOB_ID.$i.log 2>&1 &
    done
    wait
