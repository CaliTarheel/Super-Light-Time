"""Native PNG export and local Drive receipts for the guardian-closed accepted121 burial-speed continuation.

Reads accepted saved frames only. Never imports or evolves the model and never
uploads. A connector supplies actual remote IDs and metadata to record/verify.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import receipt_chain

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def shared_read(path):
    """Let the server atomically replace its checkpoint/manifest while observed."""
    if os.name != 'nt':
        return Path(path).read_bytes()
    import ctypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    create.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = create(str(Path(path).resolve()), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        return stream.read()

def read(path):
    return json.loads(shared_read(path))

def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)

def context():
    binding = read(HERE / 'binding.json')
    require(binding.get('ready') is True, 'Child delivery binding is not activated')
    require(binding.get('run_id') == receipt_chain.RUN and binding.get('status_url') == 'http://127.0.0.1:8771/api/status', 'Wrong burial child/server binding')
    require(binding.get('kind') == 'coarse_burial_native_png_drive_delivery' and binding.get('target_time_myr') == 1000, 'Wrong delivery kind/target')
    require(binding['inherited_frame_count'] == 62 and binding['first_child_epoch_myr'] == 122 and binding['public_cadence_myr'] == 2, 'Inherited boundary/cadence differs')
    require(binding['folder_id'] == 'REDACTED_DRIVE_FOLDER_ID' and binding['parent_folder_id'] == 'REDACTED_DRIVE_PARENT_ID', 'Destination differs')
    require(binding['width'] == 512 and binding['height'] == 256 and binding['native_orientation'] == dict(yaw=0,pitch=0,roll=0), 'Native format differs')
    require(sha(__file__) == binding['exporter_sha256'] and sha(HERE/'receipt_chain.py') == binding['receipt_validator_sha256'], 'Delivery source changed')
    require(sha(binding['native_encoder']) == binding['native_encoder_sha256'], 'Native encoder changed')
    records = {}
    for name in ('prepared_receipt','source_transition','control','launched_receipt','activation_receipt','resume_receipt'):
        raw = shared_read(binding[name])
        require(hashlib.sha256(raw).hexdigest() == binding[name+'_sha256'], 'Pinned '+name+' changed')
        records[name] = receipt_chain.parse(raw)
    prepared = records['prepared_receipt']; launch = records['launched_receipt']
    require(records['source_transition'] == prepared and prepared['child_run_id'] == binding['run_id'] and prepared['checkpoint_time_myr'] == 121. and prepared['frame_count'] == 62 and prepared['next_output_myr'] == 122., 'Prepared boundary differs')
    require(prepared['supported_paused_loader_verified'] is True and prepared['checkpoint_sha256'] == binding['source_boundary_checkpoint_sha256'], 'Prepared source checkpoint differs')
    require(prepared['source_revision'] == binding['source_revision'] and Path(prepared['intended_public_child']).resolve() == Path(binding['run_directory']).resolve(), 'Prepared run/source differs')
    require(records['control']['child_run_id'] == binding['run_id'] and records['control']['prepared_receipt']['sha256'] == binding['prepared_receipt_sha256'], 'Control/preparation differs')
    require(launch['kind'] == 'burial_successor_owned_interval_guardian' and launch['run_id'] == binding['run_id'] and launch['control_sha256'] == binding['control_sha256'] and launch['child_assigned_to_job'] is True and launch['job_membership_verified'] is True and launch['kill_on_job_close'] is True and launch['breakaway_allowed'] is False, 'Launch/Job binding differs')
    activation = records['activation_receipt']; resumed = records['resume_receipt']
    require(activation['run_id'] == binding['run_id'] and activation['child'] == launch['child'] and activation['guardian'] == launch['guardian'] and activation['control_sha256'] == binding['control_sha256'] and activation['prepared_sha256'] == binding['prepared_receipt_sha256'] and activation['source_revision'] == binding['source_revision'], 'Activation binding differs')
    require(resumed['run_id'] == binding['run_id'] and resumed['child'] == launch['child'] and resumed['supported_resume_calls'] == 1 and resumed['initial_interval_token'] == launch['initial_interval_token'] and resumed['initial_budget'] == launch['initial_budget'], 'Resume clock/identity differs')
    # The root source-boundary checkpoint remains immutable at121.
    require(hashlib.sha256(shared_read(Path(binding['run_directory'])/'checkpoint.npz')).hexdigest() == binding['source_boundary_checkpoint_sha256'], 'Source-boundary checkpoint changed')
    binding['_launch'] = launch
    binding['_prepared'] = prepared
    return binding

def encoder(binding):
    import numpy as np
    from PIL import Image
    tree = ast.parse(Path(binding['native_encoder']).read_bytes())
    nodes = []
    for name in ('color_image', 'png_bytes'):
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(selected) == 1 and not selected[0].decorator_list, 'Native encoder definition differs')
        nodes.extend(selected)
    namespace = dict(np=np, Image=Image, io=io)
    selected = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(selected)
    exec(compile(selected, binding['native_encoder'], 'exec'), namespace)
    return namespace['png_bytes'], np, Image

def accepted_child_rows(manifest, checkpoint_time, binding):
    inherited = binding['inherited_frame_count']
    require(inherited == 62 and binding['first_child_epoch_myr'] == 122 and binding['public_cadence_myr'] == 2, 'Child boundary differs')
    rows = manifest['frames']
    require(manifest['frame_count'] == len(rows) >= inherited, 'Accepted frame count differs')
    require(rows[inherited - 1]['time_myr'] == 120. and [r['index'] for r in rows] == list(range(len(rows))), 'Inherited history index differs')
    require(all(row['time_myr'] <= checkpoint_time for row in rows), 'Frame is newer than durable checkpoint')
    future = rows[inherited:]
    require([r['time_myr'] for r in future] == list(range(122, 122 + 2 * len(future), 2)), 'Accepted future cadence differs')
    require(all(row['time_myr'] <= binding['target_time_myr'] for row in future), 'Child frame exceeds bound target')
    return future

def export(binding):
    import numpy as np
    run = Path(binding['run_directory'])
    selected = receipt_chain.select(binding, binding['_launch'], shared_read, lambda p: Path(p).is_file())
    checkpoint_raw = shared_read(selected['checkpoint']['path'])
    require(hashlib.sha256(checkpoint_raw).hexdigest() == selected['checkpoint']['sha256'], 'Latest accepted checkpoint hash differs')
    with zipfile.ZipFile(io.BytesIO(checkpoint_raw)) as archive:
        metadata_array = np.load(io.BytesIO(archive.read('__checkpoint_json.npy')), allow_pickle=False)
    require(metadata_array.dtype == np.uint8 and metadata_array.ndim == 1, 'Checkpoint metadata type differs')
    metadata_raw = metadata_array.tobytes()
    checkpoint_header = json.loads(metadata_raw)
    prepared = binding.get('_prepared') or read(binding['prepared_receipt'])
    require(checkpoint_header['version'] == 1 and checkpoint_header['compatibility'] == prepared['compatibility'],
            'Durable checkpoint scientific source/backends differ')
    manifest = checkpoint_header['manifest']
    require(manifest['run_id'] == binding['run_id'] and manifest['source_commit'] == binding['source_revision'],
            'Saved run/source identity differs')
    require(checkpoint_header['time_myr'] == selected['accepted_checkpoint_myr'], 'Accepted checkpoint time differs')
    rows = manifest['frames']
    future = accepted_child_rows(manifest, checkpoint_header['time_myr'], binding)
    require([(r['index'], r['time_myr']) for r in future] == [(r['index'], r['time_myr']) for r in selected['frames']], 'Checkpoint history differs from closed intervals')
    frame_pins = {r['index']:r for r in selected['frames']}
    png_bytes, np, Image = encoder(binding)
    output = Path(binding['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in future:
        stem = run / f"frame_{row['index']:04d}"
        frame_metadata_raw = shared_read(stem.with_suffix('.json'))
        frame_metadata = receipt_chain.parse(frame_metadata_raw)
        require(all(frame_metadata[key] == row[key] for key in ('index', 'time_myr')), 'Saved frame metadata differs')
        require(frame_metadata['width'] == 512 and frame_metadata['height'] == 256, 'Native raster resolution changed')
        raw = shared_read(stem.with_suffix('.npz'))
        frame_sha = hashlib.sha256(raw).hexdigest()
        metadata_sha = hashlib.sha256(frame_metadata_raw).hexdigest()
        require(frame_sha == frame_pins[row['index']]['npz']['sha256'] and metadata_sha == frame_pins[row['index']]['json']['sha256'], 'Guardian-closed frame hash differs')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(archive.testzip() is None, 'Saved frame CRC failed')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            elevation = archive['elevation'].copy()
        require(elevation.size == 512 * 256 and np.isfinite(elevation).all(), 'Saved elevation shape or finite check failed')
        frame = dict(frame_metadata, elevation=elevation)
        for colored, suffix in ((False, 'heightmap16'), (True, 'relief')):
            payload = png_bytes(frame, colored=colored)
            require(payload[:8] == b'\x89PNG\r\n\x1a\n' and payload[24:26] == bytes((8, 2) if colored else (16, 0)),
                    'Native PNG bit depth/color type differs')
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                require(image.size == (512, 256) and image.format == 'PNG', 'PNG dimensions/format differ')
                if not colored:
                    expected = np.rint(np.clip(elevation.astype(float) + 12000., 0., 65535.)).astype(np.uint16).reshape(256, 512)
                    require(np.array_equal(np.asarray(image), expected), 'Fixed-scale heightmap pixels differ')
                else:
                    require(image.mode == 'RGB', 'Relief mode differs')
            name = f"Asein_Lite_{binding['run_id']}_{int(row['time_myr']):04d}Myr_{suffix}.png"
            path = output / name
            if path.exists():
                require(path.read_bytes() == payload, 'Existing immutable PNG bytes differ')
            else:
                with path.open('xb') as stream:
                    stream.write(payload)
            entries.append(dict(path=str(path), name=name, bytes=len(payload), sha256=sha(path), mime_type='image/png',
                frame_index=row['index'], elapsed_myr=row['time_myr'], frame_sha256=frame_sha,
                frame_metadata_sha256=metadata_sha, png_kind=suffix, width=512, height=256,
                png_bit_depth=8 if colored else 16, elevation_decode=None if colored else 'metres = uint16_pixel - 12000'))
        require(hashlib.sha256(shared_read(stem.with_suffix('.npz'))).hexdigest() == frame_sha and hashlib.sha256(shared_read(stem.with_suffix('.json'))).hexdigest() == metadata_sha,
                'Frame changed during export; retry later')
    again = receipt_chain.select(binding, binding['_launch'], shared_read, lambda p: Path(p).is_file())
    require(again == selected, 'Acceptance chain changed during export; retry later')
    require(hashlib.sha256(shared_read(selected['checkpoint']['path'])).hexdigest() == selected['checkpoint']['sha256'], 'Selected checkpoint changed during export')
    result = dict(kind='coarse_burial_guardian_closed_native_png_export', version=1, run_id=binding['run_id'],
        source_revision=binding['source_revision'], accepted_frame_count=len(rows), inherited_frame_count=62,
        png_count=len(entries), files=entries, native_encoder_sha256=binding['native_encoder_sha256'],
        accepted_checkpoint_myr=checkpoint_header['time_myr'],
        checkpoint_metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
        acceptance=selected, verification_scope=dict(all_small_receipts_rehashed=True, latest_checkpoint_full_hash_verified=True, source_boundary_full_hash_verified=True, all_exported_child_frame_hashes_verified=True, historical_checkpoint_digest_scope='Carried from independently guardian-verified immutable receipts; historical checkpoint bytes are not rehashed on each poll.'),
        model_imported=False, model_stepped=False, remote_uploads=0, updated_at_utc=datetime.now(timezone.utc).isoformat())
    write(binding['export_receipt'], result)
    print(json.dumps(dict(state='exported' if entries else 'await_first122', accepted_frame_count=len(rows),
        child_png_count=len(entries), receipt=binding['export_receipt'], receipt_sha256=sha(binding['export_receipt']))))

def sync(binding):
    path = Path(binding['sync_receipt'])
    result = read(path) if path.exists() else dict(kind='coarse_rigid_sheet_drive_png_sync', version=1,
        run_id=binding['run_id'], folder_id=binding['folder_id'], files=[])
    require(result['run_id'] == binding['run_id'] and result['folder_id'] == binding['folder_id'], 'Sync identity differs')
    return result

def pending(binding):
    exported = read(binding['export_receipt'])
    require(exported['run_id'] == binding['run_id'], 'Export identity differs')
    known = {item['path']: item for item in sync(binding)['files']}
    result = []
    for item in exported['files']:
        require(sha(item['path']) == item['sha256'], 'Exported local PNG changed')
        prior = known.get(item['path'])
        if prior:
            require(prior['sha256'] == item['sha256'], 'Recorded uploaded bytes differ')
        if not prior or not prior.get('verified'):
            result.append(dict(item, drive_id=prior.get('id') if prior else None,
                               action='verify_existing_id' if prior else 'check_destination_name_then_upload'))
    print(json.dumps(dict(run_id=binding['run_id'], folder_id=binding['folder_id'], pending=result,
                         verified_png_count=sum(bool(row.get('verified')) for row in known.values()))))

def record(binding, input_file, verify=False):
    data = read(input_file)
    exported = read(binding['export_receipt'])
    matches = [row for row in exported['files'] if row['path'] == data['path']]
    require(len(matches) == 1, 'Remote record must name an exact exported child PNG')
    item = matches[0]
    require(sha(item['path']) == item['sha256'], 'Local uploaded PNG bytes changed')
    require(isinstance(data['id'], str) and data['id'], 'Actual Drive id required')
    result = sync(binding)
    previous = [row for row in result['files'] if row['path'] == item['path']]
    require(not previous or previous[0]['id'] == data['id'], 'Different Drive id already recorded')
    require(not verify or previous, 'Record returned upload id before verifying metadata')
    if verify:
        require(data['name'] == item['name'] and data['mime_type'] == 'image/png'
                and int(data['size']) == item['bytes'] and binding['folder_id'] in data['parents']
                and isinstance(data.get('url'), str) and data['url'], 'Drive metadata readback differs')
    entry = dict(item, id=data['id'], url=data.get('url'), verified=bool(verify or (previous and previous[0].get('verified'))))
    if verify:
        entry['verified_metadata'] = data
        entry['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
    elif previous and previous[0].get('verified'):
        entry = previous[0]
    result['files'] = [row for row in result['files'] if row['path'] != item['path']] + [entry]
    result['verified_png_count'] = sum(bool(row.get('verified')) for row in result['files'])
    result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(binding['sync_receipt'], result)
    print(json.dumps(dict(state='verified' if entry['verified'] else 'uploaded_unverified', id=data['id'],
        receipt=binding['sync_receipt'], receipt_sha256=sha(binding['sync_receipt']))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('export', 'pending', 'record', 'verify'))
    parser.add_argument('--input-file')
    args = parser.parse_args()
    binding = context()
    if args.command == 'export':
        export(binding)
    elif args.command == 'pending':
        pending(binding)
    else:
        require(args.input_file, 'Connector metadata JSON file required')
        record(binding, args.input_file, verify=args.command == 'verify')
