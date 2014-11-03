Batch Submission
================

Multiple ``osprey worker`` processes can be run simultaneously and connect
to the same trials database. The following scripts might be useful as templates
for submitting multiple parallel ``osprey worker`` s to a cluster batch scheduling
system.

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

Example SLURM Script
--------------------
