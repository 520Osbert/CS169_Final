import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class JSSP:
    '''
    m machines, n jobs, each machine need to finish jobs with specific orders
    '''
    def __init__(self, n, m, Processing_time):
        self.m_machine = m
        self.n_job = n
        self.Processing_time = Processing_time
        assert self.Processing_time.shape == (self.n_job, self.m_machine)


    def get_end_time(self, Seq):
        '''
        Seq is nxm that each column lists the jobs that machine will do in order
        '''
        assert Seq.shape == (self.n_job, self.m_machine)
        machine_end_time = np.zeros(self.m_machine)
        job_end_time = np.zeros(self.n_job)
        for i in range(self.n_job):
            for j in range(self.m_machine):
                job = int(Seq[i, j])
                print(job, j)
                end = max(machine_end_time[j], job_end_time[job]) + self.Processing_time[job, j]
                machine_end_time[j] = end
                job_end_time[job] = end
                print(machine_end_time, job_end_time)
        return np.max(machine_end_time)

    def schedule_efficiency(self, Seq):
        '''
        Scheduling efficiency can be defined for a schedule through the ratio
        of total machine idle time to the total processing time
        '''
        end_time = self.get_end_time(Seq)
        total_processing_time = end_time * self.m_machine
        idle_time = total_processing_time - np.sum(self.Processing_time)
        return idle_time / total_processing_time


    def plot(self, Seq):
        fig, ax = plt.subplots()
        cmap = plt.cm.get_cmap("summer", self.n_job)(np.linspace(0.15, 0.85, self.n_job))

        machine_end_time = np.zeros(self.m_machine)
        job_end_time = np.zeros(self.n_job)
        for i in zip(range(self.n_job)):
            for j in range(self.m_machine):
                job = int(Seq[i, j])
                end = max(machine_end_time[j], job_end_time[job]) + self.Processing_time[job, j]
                ax.barh(j, color=cmap[job], width=self.Processing_time[job, j], left=max(machine_end_time[j], job_end_time[job]))
                machine_end_time[j] = end
                job_end_time[job] = end

        blocks = [mpatches.Patch(color=cmap[i], label="Job "+str(i)) for i in range(self.n_job)]
        ax.legend(handles=blocks, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_yticks(range(self.m_machine))
        ax.set_yticklabels(['Machine ' + str(i + 1) for i in range(self.m_machine)])
        ax.set_xlabel("time")
        plt.show()
        return fig