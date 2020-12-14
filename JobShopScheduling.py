import numpy as np

class JSSP:
    '''
    m machines, n jobs, each machine need to finish jobs with specific orders
    '''
    def __init__(self, n, m, cost_matrix):
        self.m_machine = m
        self.n_job = n
        self.Cost = cost_matrix
        assert self.Cost.shape() == (self.m_machine, self.n_job)

    def check_valid(self, Seq):
        '''
        Check if the sequence is valid: No overlapping
        '''
        return True

    def total_processing_time(self, Seq):
        '''
        Seq is mxn that specifies the start time to finish jobs for each machine
        Assert start time i + process time i <= start time i+1
        '''
        assert Seq.shape == (self.m_machine, self.n_job)
        if not self.check_valid(Seq):
            return np.inf
        last_end_job = np.argmax(Seq, axis=1)
        last_end_time = np.max(Seq, axis=1)
        end_time = np.max([last_end_time[i] + self.Cost[i, last_end_job[i]] for i in range(self.m_machine)])
        return end_time

    def schedule_efficiency(self, Seq):
        '''
        Scheduling efficiency can be defined for a schedule through the ratio
        of total machine idle time to the total processing time
        '''
        pass