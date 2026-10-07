"""Real spawn-process coverage/inverses and validated local control transport."""
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from parallel_runtime import (RuntimePool, active_runtime, physical_core_count,
                              share_inputs, read_inputs, worker_probe)
import numpy as np
import mesh_coverage
from mesh_geometry import icosphere, build_locator
import viscous_sheet as sheet


def read_only_sum(spec):
    with read_inputs(spec) as values:
        immutable = not values['array'].flags.writeable
        return float(values['array'].sum()), immutable


class ParallelRuntimeTests(unittest.TestCase):
    def test_shared_arrays_are_read_only_and_unlinked_after_batch(self):
        from multiprocessing import shared_memory
        with RuntimePool(workers=1) as runtime:
            with share_inputs({'array': np.arange(12.)}) as spec:
                name = spec['array'].name
                self.assertEqual(runtime.map(read_only_sum, [spec]), [(66., True)])
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
        self.assertIsNone(active_runtime())

    def test_worker_settings_and_reusable_pool(self):
        count = min(2, physical_core_count())
        with RuntimePool(workers=count, max_workers=count) as runtime:
            first = runtime.map(worker_probe, [.15]*count)
            second = runtime.map(worker_probe, [.15]*count)
            self.assertEqual({r['pid'] for r in first}, {r['pid'] for r in second})
            for row in first+second:
                self.assertTrue(all(v == '1' for v in row['blas_environment'].values()))
                if row['priority_class'] is not None:
                    self.assertEqual(row['priority_class'], 0x4000)
            self.assertEqual(runtime.status()['completed_batches'], 2)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            runtime.map(worker_probe, [0.])

    def test_worker_error_drains_batch_releases_shared_inputs_and_keeps_pool_usable(self):
        from multiprocessing import shared_memory
        with RuntimePool(workers=min(2,physical_core_count())) as runtime:
            with share_inputs({'array':np.arange(7.)}) as shared:
                name=shared['array'].name
                with self.assertRaises(KeyError):
                    runtime.map(read_only_sum,[shared,{},shared])
                self.assertEqual(runtime.policy.status()['in_flight'],0)
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
            self.assertTrue(runtime.map(worker_probe,[0.])[0]['pid'])

    def test_ordered_coverage_is_bitwise_identical_on_rotated_mesh(self):
        mesh = icosphere(4)
        source = icosphere(4)
        angle = .037
        rotation = np.array([[np.cos(angle),-np.sin(angle),0.],
                             [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        vertices = source['vertices']@rotation.T
        original = vertices.copy()
        locator = build_locator(mesh['vertices'], mesh['faces'])
        expected = mesh_coverage.intersections(mesh,vertices,source['faces'],locator=locator)
        count = min(2, physical_core_count())
        with RuntimePool(workers=count, max_workers=count) as runtime:
            actual = mesh_coverage.intersections(mesh,vertices,source['faces'],locator=locator)
            if count > 1:
                self.assertEqual(runtime.status()['completed_batches'], 1)
        for key in ('material_index','control_index','area_km2'):
            np.testing.assert_array_equal(actual[key],expected[key])
        self.assertEqual(actual['candidate_pairs'],expected['candidate_pairs'])
        np.testing.assert_array_equal(vertices,original)

    def test_small_control_mesh_bypasses_process_pool(self):
        mesh=icosphere(2);source=icosphere(4)
        expected=mesh_coverage.intersections(mesh,source['vertices'],source['faces'])
        with RuntimePool(workers=min(2,physical_core_count())) as runtime:
            actual=mesh_coverage.intersections(mesh,source['vertices'],source['faces'])
            self.assertFalse(runtime.status()['pool_started'])
            self.assertEqual(runtime.status()['completed_batches'],0)
        for key in ('material_index','control_index','area_km2'):
            np.testing.assert_array_equal(actual[key],expected[key])

    def test_independent_inverse_is_bitwise_exact_with_variable_viscosity(self):
        mesh = icosphere(1)
        points,faces,area = mesh['vertices'],mesh['faces'],mesh['area_km2']
        weights = np.linspace(.4,1.3,len(faces))
        context = sheet.prepare(points,faces,area,100.,viscosity_weights=weights)
        rigid = np.zeros(len(points),bool);rigid[0] = True
        node = 3
        axis = np.eye(3)[np.argmin(np.abs(points[node]))]
        first = np.cross(points[node],axis);first /= np.linalg.norm(first)
        vectors = (first,np.cross(points[node],first))
        stage = dict(points=points,faces=faces,area=area,data=context['data'],rigid=rigid,
                     length_km=100.,radius=6371.,viscosity_weights=weights,
                     iterations=1024,tolerance=1e-10)
        reference = []
        for vector in vectors:
            rhs = np.zeros_like(points);rhs[node] = vector
            body = np.divide(rhs,context['data'][:,None],out=np.zeros_like(rhs),where=context['data'][:,None]>0)
            reference.append(sheet.solve(points,faces,area,np.zeros_like(points),rigid,
                np.zeros(len(points),bool),100.,body_force=body,viscosity_weights=weights,
                iterations=1024,tolerance=1e-10))
        with RuntimePool(workers=min(2,physical_core_count())) as runtime:
            with share_inputs(stage) as shared:
                actual = runtime.map(sheet.independent_inverse,[(shared,node,vector) for vector in vectors])
        for (value,info),(expected,diagnostics) in zip(actual,reference):
            self.assertTrue(info['converged'])
            np.testing.assert_array_equal(value,expected)
            self.assertEqual(info,diagnostics)

    def test_constrained_gravity_prefetch_preserves_endpoint_and_diagnostics(self):
        import gravitational_relaxation as gravity
        from material_surface import spherical_face_areas
        xy=np.array([[100.,0.],[0.,100.],[-100.,0.],[0.,-100.],[0.,0.]])
        points=np.column_stack((xy/6371.,np.ones(5)))
        points/=np.linalg.norm(points,axis=1)[:,None]
        faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        area=spherical_face_areas(points,faces,6371.)
        volume=area*np.array([75.,30.,70.,30.])
        args=(points,faces,volume,np.array([42.,8.,8.,30.]),np.ones(4,int),2.)
        kwargs=dict(rigid_mask=np.array([True,True,True,True,False]),
            minimum_area_km2=volume/75.,maximum_area_km2=volume/8.,
            tolerance=1e-8,iterations=1024,constraint_version=1)
        reference=gravity.relax(*args,**kwargs)
        count=min(2,physical_core_count())
        with RuntimePool(workers=count,max_workers=count) as runtime:
            actual=gravity.relax(*args,**kwargs)
            if count>1:self.assertGreater(runtime.status()['completed_batches'],0)
        np.testing.assert_array_equal(actual[0],reference[0])
        self.assertEqual(actual[1],reference[1])

    def test_local_api_validates_origin_shape_and_worker_limit(self):
        from http.server import ThreadingHTTPServer
        from server import Handler, SimulationManager
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(Path(directory),workers=1)
            class TestHandler(Handler):
                pass
            TestHandler.manager = manager
            server = ThreadingHTTPServer(('127.0.0.1',0),TestHandler)
            thread = Thread(target=server.serve_forever,daemon=True);thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def request(body=None,origin=None):
                headers={'Content-Type':'application/json'}
                if origin is not None:headers['Origin']=origin
                req=Request(base+'/api/workers',data=None if body is None else json.dumps(body).encode(),
                            headers=headers,method='GET' if body is None else 'POST')
                with urlopen(req,timeout=5) as response:
                    return json.loads(response.read())
            try:
                self.assertEqual(request()['requested_workers'],1)
                maximum=request()['max_workers']
                self.assertEqual(request({'workers':maximum})['requested_workers'],maximum)
                for body in ({'workers':True},{'workers':0},{'workers':maximum+1},
                             {'workers':1,'dt_myr':1},{'mode':'bad'},[]):
                    with self.assertRaises(HTTPError) as error:request(body)
                    self.assertEqual(error.exception.code,400)
                with self.assertRaises(HTTPError) as error:request({'workers':1},'https://example.org')
                self.assertEqual(error.exception.code,403)
                self.assertEqual(request()['requested_workers'],maximum)
                self.assertEqual(request({'workers':1},base)['requested_workers'],1)
                self.assertFalse(manager.parallel.status()['pool_started'])
            finally:
                server.shutdown();thread.join();server.server_close();manager.parallel.close()


if __name__ == '__main__':
    unittest.main()
