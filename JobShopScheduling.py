import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class JSSP:
    '''
    m machines, n jobs, each machine need to finish jobs with specific orders
    '''
    def __init__(self, n, m, Processing_time=0, randopt=False, low=1, high=20):
        self.m_machine = m
        self.n_job = n
        if randopt:
            self.Processing_time = self.generate_rand_proc(low, high)
        else:
            self.Processing_time = Processing_time
        assert self.Processing_time.shape == (self.n_job, self.m_machine)

    def generate_rand_proc(self, low, high):
        return np.random.randint(low, high, size=(self.n_job, self.m_machine))

    def generate_rand_seq(self):
        return np.array([np.random.permutation(self.m_machine) for _ in range(self.n_job)])


    def get_end_time(self, Seq):
        '''
        Seq is nxm that each row lists the machines that the job will do in order
        '''
        assert Seq.shape == (self.n_job, self.m_machine)
        machine_end_time = np.zeros(self.m_machine)
        job_end_time = np.zeros(self.n_job)
        for j in range(self.m_machine):
            for i in range(self.n_job):
                machine = int(Seq[i, j])
                print(machine, i)
                end = max(machine_end_time[machine], job_end_time[i]) + self.Processing_time[i, j]
                machine_end_time[machine] = end
                job_end_time[i] = end
                print(machine_end_time, job_end_time, end)
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
        for j in range(self.m_machine):
            for i in range(self.n_job):
                machine = int(Seq[i, j])
                end = max(machine_end_time[machine], job_end_time[i]) + self.Processing_time[i, j]
                ax.barh(machine, color=cmap[i], width=self.Processing_time[i, j],
                        left=max(machine_end_time[machine], job_end_time[i]))

                machine_end_time[machine] = end
                job_end_time[i] = end

        blocks = [mpatches.Patch(color=cmap[i], label="Job "+str(i)) for i in range(self.n_job)]
        ax.legend(handles=blocks, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_yticks(range(self.m_machine))
        ax.set_yticklabels(['Machine ' + str(i + 1) for i in range(self.m_machine)])
        ax.set_xlabel("time")
        plt.show()
        return fig