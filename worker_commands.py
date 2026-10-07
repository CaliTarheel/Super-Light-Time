"""Performance-only command facade for the next run's local server.

The caller owns HTTP transport and existing local Host/Origin validation.
This module does not change physics configuration or create worker processes.
"""
from __future__ import annotations

VERSION = 1


class WorkerCommands:
    def __init__(self, policy):
        self.policy = policy

    def status(self):
        state = self.policy.status()
        return dict(version=VERSION, **state,
                    draining=state['in_flight'] > state['requested_workers'],
                    scope='Maximum concurrent parallel work tasks; not CPU affinity or a utilization guarantee.')

    def apply(self, body):
        if not isinstance(body, dict) or len(body) != 1:
            raise ValueError('Specify exactly one workers count or mode.')
        if 'workers' in body:
            requested = body['workers']
        elif 'mode' in body:
            maximum = self.policy.status()['max_workers']
            modes = {'quiet': min(2, maximum), 'maximum': maximum}
            if not isinstance(body['mode'], str) or body['mode'] not in modes:
                raise ValueError('Worker mode must be quiet or maximum.')
            requested = modes[body['mode']]
        else:
            raise ValueError('Unknown worker-control field.')
        self.policy.set_limit(requested)
        return self.status()
