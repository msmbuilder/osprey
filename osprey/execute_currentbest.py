from __future__ import print_function

import numpy as np
from .config import Config
from .trials import Trial

def execute(args, parser):
    config = Config(args.config, verbose=False)

    session = config.trials()


    items = [curr.to_dict() for curr in session.query(Trial).all()]
    if not items:
        print('No Models Found')
    else:
        c_b_m = items[np.argmax([i["mean_test_score"] for i in items])]
        parameter_dict = c_b_m["parameters"]
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Best Current Model  = %f+-%f' %(c_b_m["mean_test_score"],\
                                                np.std(c_b_m["test_scores"])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('PipelineStep\tParamter \t Value')
        for i in config.estimator().steps:
            print(i[0])
            for param in sorted(parameter_dict.keys()):
                if str(param).startswith(i[0]):
                    print("\t\t",param.split("__")[1],"\t",parameter_dict[param])

    return