from pyiron_workflow import Workflow, as_function_node
from pyiron import Project


@as_function_node("project")
def CreateProject(name: str):
    return Project(name)

@as_function_node("createjob")
def Lammps_md(project: Project, structure, potential: str, job_name: str = 'job',
          delete_existing_job: bool = True, temperature: int = 300,
             n_steps: int = 100, time_step: int = 100):    
    job = project.create.job.Lammps(job_name, delete_existing_job = delete_existing_job)
    job.structure = structure
    job.potential = potential
    job.calc_md(temperature= temperature, n_ionic_steps= n_steps, time_step= time_step)
    job.run(delete_existing_job = True)
    return(job)