def stopping_at(tolerance, maximize=False, verbose=True):
    """Create a callback that activates early stoppping.
    The difference between the best and the second best
    validation errors over the last one needs to be greater 
    than the assigned tolerance to continue training.
    Requires at least one item in evals.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    bst.best_score, bst.best_iteration and bst.best_ntree_limit.
    (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
    and/or num_class appears in the parameters)
    Parameters
    ----------
    tolerance : float
       The tolerance threshold.
    maximize : bool
        Whether to maximize evaluation metric.
    verbose : optional, bool
        Whether to print message about early stopping information.
    Returns
    -------
    callback : function
        The requested callback function.
    """
    import xgboost as xgb
    import numpy as np
    state = {}
    loss = []
    def init(env):
        """internal function"""
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For early stopping you need at least one set in evals.')
        if len(env.evaluation_result_list) > 1 and verbose:
            msg = ("Multiple eval metrics have been passed: "
                   "'{0}' will be used for early stopping.\n\n")
            xgb.rabit.tracker_print(msg.format(env.evaluation_result_list[-1][0]))
        maximize_metrics = ('auc', 'map', 'ndcg')
        maximize_at_n_metrics = ('auc@', 'map@', 'ndcg@')
        maximize_score = maximize
        metric = env.evaluation_result_list[-1][0]

        if any(env.evaluation_result_list[-1][0].split('-')[-1].startswith(x)
               for x in maximize_at_n_metrics):
            maximize_score = True

        if any(env.evaluation_result_list[-1][0].split('-')[-1].split(":")[0] == x
               for x in maximize_metrics):
            maximize_score = True

        if verbose and env.rank == 0:
            msg = "Will train until {} hasn't improved more than {}.\n"
            xgb.rabit.tracker_print(msg.format(metric, tolerance))

        state['maximize_score'] = maximize_score
        state['best_iteration'] = 0
        if maximize_score:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')
        
        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        score = env.evaluation_result_list[-1][1]
        if len(state) == 0:
            init(env)
        loss.append(env.evaluation_result_list[-1][1])
        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        sort = sorted(loss)
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d]\t%s' % (
                env.iteration,
                '\t'.join([str(x) for x in env.evaluation_result_list]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        if len(sort) > 2:
            if ((np.abs(sort[-1] - sort[-2]) / loss[-1]) < tolerance and not maximize_score) or \
            ((np.abs(sort[0] - sort[1]) / loss[-1]) < tolerance and maximize_score):
                raise xgb.core.EarlyStopException(best_iteration)

    return callback