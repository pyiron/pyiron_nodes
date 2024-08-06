import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = "vasp.log"
handlers = [VaspErrorHandler(output_filename=output_filename), UnconvergedErrorHandler(), NonConvergingErrorHandler(), PositiveEnergyErrorHandler()]
jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix = ".relax_1", final=False, settings_override=[{"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5}}}]),
        VaspJob(sys.argv[1:], output_file=output_filename, suffix = ".relax_2", final=False,
                settings_override = [{"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
                    {"dict": "INCAR", "action": {"_set": {"KSPACING": 0.5, "EDIFF": 1E-5, "EDIFFG": 1E-4}}}], copy_magmom=True),
        VaspJob(sys.argv[1:], output_file=output_filename, suffix = "",
                settings_override = [{"dict": "INCAR", "action": {"_set": {"NSW": 0, "LAECHG": True, "LCHARGE": True, "NELM": 240, "EDIFF": 1E-5}}},
                                     {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}])]
c = Custodian(handlers, jobs, max_errors=10)
c.run()
