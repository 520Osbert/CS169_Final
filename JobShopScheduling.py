import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Particle

class JSSP:
    '''
    m machines, n jobs, each machine need to finish jobs with specific orders
    '''
    def __init__(self, n, m, Processing_time, randopt=False, low=1, high=20):
        self.m_machine = m
        self.n_job = n
        self.Processing_time = Processing_time
        if randopt:
            self.Processing_time = self.generate_rand_proc(low, high)
        
        assert self.Processing_time.shape == (self.n_job, self.m_machine)

    def generate_rand_proc(self, low, high):
        return np.random.randint(low, high, size=(self.n_job, self.m_machine))

    def generate_rand_seq(self):
        return np.array([np.random.permutation(self.m_machine) for _ in range(self.n_job)])

    def generate_rand_velocity(self):
        return np.random.uniform(low=-(self.m_machine-1), high=self.m_machine-1, size=(self.n_job,self.m_machine))
    
    def PSO(self, particle_count):
        '''
        Use Particle Swarm Optimization to find minimal makespan given particle_count particles
        '''
        particles = [Particle.particle(self.generate_rand_seq(), self.generate_rand_velocity()) for _ in range(particle_count)]
        retval = Particle.particle_swarm_optimization(self.get_end_time, particles, 200)
        return retval

    def get_end_time(self, Seq, verbose=False):
        '''
        Seq is nxm that each row lists the machines that the job will do in order
        '''
        assert Seq.shape == (self.n_job, self.m_machine), f"{Seq.shape}, {self.n_job}, { self.m_machine}"
        machine_end_time = np.zeros(self.m_machine)
        job_end_time = np.zeros(self.n_job)
        for j in range(self.m_machine):
            for i in range(self.n_job):
                machine = int(Seq[i, j])
                if verbose:
                    print(machine, i)
                end = max(machine_end_time[machine], job_end_time[i]) + self.Processing_time[i, j]
                machine_end_time[machine] = end
                job_end_time[i] = end
                if verbose:
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
        assert Seq.shape == (self.n_job, self.m_machine)
        Seq = Seq.astype(int)
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