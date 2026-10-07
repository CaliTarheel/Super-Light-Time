"""Loopback API contracts for local landscape import and derived old margins."""
import json
import sys
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server as app_server


class ImportRoutes(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        class Jobs:
            def start_gospl_result(self, path, **kwargs):
                calls.append((path, kwargs))
                return dict(job_id='landscape', state='running', source_type='gospl_result')
            def start(self, manager, index, **kwargs):
                calls.append((index, kwargs))
                return dict(job_id='native', state='running')
        class Handler(app_server.Handler):
            manager = SimpleNamespace(shutting_down=False)
            terrain_manager = Jobs()
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(5)

    def post(self, route, body, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        request = Request(self.base + route, data=json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_import_uses_selected_epoch_and_keeps_result_orientation(self):
        response = self.post('/api/terrain/gospl', dict(path='C:/Worlds/output', epoch_index=20,
                             width=4096, orientation=dict(yaw=90)))
        self.assertEqual(response['source_type'], 'gospl_result')
        self.assertEqual(self.calls, [('C:/Worlds/output', dict(epoch_index=20, width=4096, detail=0.))])

    def test_inspection_browse_cancel_and_read_errors_are_reviewable(self):
        with patch.object(app_server, 'inspect_result', return_value=dict(default_epoch=20, epochs=[dict(index=20)])):
            self.assertEqual(self.post('/api/gospl/results/inspect', {'path': 'output'})['default_epoch'], 20)
        with patch.object(app_server, 'choose_gospl_result', return_value=None):
            self.assertEqual(self.post('/api/gospl/results/browse', {}), {'path': None})
        with patch.object(app_server, 'inspect_result', side_effect=ValueError('Incomplete result: missing partition.')):
            with self.assertRaises(HTTPError) as raised:
                self.post('/api/gospl/results/inspect', {'path': 'output'})
            self.assertEqual(raised.exception.code, 400)
            self.assertIn('missing partition', json.load(raised.exception)['error'])

    def test_old_margin_option_is_explicit_and_cross_origin_cannot_open_picker(self):
        self.post('/api/terrain', {'frame': 94, 'reconstruct_margins': True})
        self.assertTrue(self.calls[0][1]['reconstruct_margins'])
        self.post('/api/terrain', {'frame': 94})
        self.assertFalse(self.calls[1][1]['reconstruct_margins'])
        with patch.object(app_server, 'choose_gospl_result') as picker:
            with self.assertRaises(HTTPError) as raised:
                self.post('/api/gospl/results/browse', {}, origin='https://example.com')
            self.assertEqual(raised.exception.code, 403)
            picker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
