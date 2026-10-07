"""New surface profiles are reproducible from the exported helper closure."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import continental_margin
import gospl_export
import native_frame_sampling
from tests.test_continental_margin import source_frame
from tests.test_native_gospl_sampling import write_frame
from validate_gospl import validate_package


class SourceClosureTests(unittest.TestCase):
    def test_recursive_capture_follows_helper_imports_and_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'native_frame_sampling.py').write_text('import first\nimport numpy\n')
            (folder/'first.py').write_text('from second import value\n')
            (folder/'second.py').write_text('import first\nvalue=7\n')
            (folder/'unrelated.py').write_text('not_a_helper=1\n')
            self.assertEqual(set(gospl_export.capture_native_sampling_sources(folder)),
                {'native_frame_sampling.py','first.py','second.py'})

    def test_new_margin_package_preserves_surface_contract_and_standalone_sampling(self):
        first=continental_margin.upgrade_frame(source_frame(),width_km=130.,shelf_depth_m=160.)
        second=deepcopy(first);second['time_myr']=2.;second['material_height_m']+=70.
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);run=folder/'run';run.mkdir()
            for index,frame in enumerate((first,second)):write_frame(run,index,frame)
            manifest=dict(run_id='margin-source-closure',config={'dt_myr':2.},
                          frames=[{'time_myr':0.},{'time_myr':2.}])
            output=folder/'export'
            metadata=gospl_export.build_history(run,output,manifest,subdivisions=2,dt_years=100000)
            self.assertTrue(validate_package(output)['passed'])
            for row in metadata['source_frames']:
                contract=row['surface_reconstruction']
                self.assertEqual(contract['continental_margin_version'],1)
                self.assertEqual(contract['continental_margin_parameters'],{'width_km':130.,'shelf_depth_m':160.})
                self.assertEqual(contract['arc_surface_version'],2)
            for name in ('continental_margin.py','continental_contacts.py'):
                self.assertEqual(hashlib.sha256((output/'source'/name).read_bytes()).hexdigest(),
                    metadata['native_sampling_sources_sha256'][name])
            with np.load(output/'input/mesh.npz') as mesh:
                expected=native_frame_sampling.sample_frame(first,mesh['v']/gospl_export.RADIUS_M)['elevation']
                np.testing.assert_allclose(mesh['z'],expected,atol=.0005)
            # Import and sample from only the captured package sources.
            np.savez(output/'source/probe-frame.npz',**{k:v for k,v in first.items() if isinstance(v,np.ndarray)})
            (output/'source/probe-frame.json').write_text(json.dumps({k:v for k,v in first.items() if not isinstance(v,np.ndarray)}))
            program="""import sys;sys.path.insert(0,'.')
import json,numpy as np,native_frame_sampling as s,gospl_export as e
f=json.load(open('probe-frame.json'));a=np.load('probe-frame.npz');f.update({k:a[k] for k in a.files})
m=np.load('../input/mesh.npz');z=s.sample_frame(f,m['v']/e.RADIUS_M)['elevation']
assert np.max(abs(z-m['z']))<.0005
assert 'continental_contacts.py' in e.NATIVE_SAMPLING_SOURCES
"""
            subprocess.run([sys.executable,'-I','-c',program],cwd=output/'source',check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
