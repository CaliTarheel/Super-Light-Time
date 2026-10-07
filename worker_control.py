"""Chat-accessible CLI client for the future local /api/workers route.

No local override file or success is fabricated if the route is unavailable.
"""
from __future__ import annotations
import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group()
    command.add_argument('--workers', type=int)
    command.add_argument('--mode', choices=['quiet', 'maximum'])
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme != 'http' or parsed.hostname not in ['127.0.0.1', 'localhost', '::1'] or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ['', '/']:
        parser.error('Use the local Deep Time server base URL.')
    body = None
    if args.workers is not None: body = {'workers': args.workers}
    elif args.mode is not None: body = {'mode': args.mode}
    request = Request(args.url.rstrip('/')+'/api/workers',
        data=None if body is None else json.dumps(body).encode('utf-8'),
        headers={'Content-Type':'application/json'}, method='GET' if body is None else 'POST')
    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
        if not isinstance(result, dict) or result.get('version') != 1 or not all(k in result for k in ['max_workers','requested_workers','in_flight']):
            raise ValueError('Server did not return a supported worker-control status.')
    except (HTTPError, URLError, ValueError) as error:
        parser.exit(1, f'Worker control unavailable or rejected: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
