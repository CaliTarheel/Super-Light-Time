"""Narrow HDF5-to-NPY bridge; also runnable in the installed goSPL environment.

Only explicit numeric datasets are read. External/soft links and virtual
datasets are refused; paths, shapes and output files are supplied by the trusted
importer, never evaluated as shell or XML expressions.
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np


def extract(requests, output_dir, check_cancel=None):
    import h5py
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    result=[]
    for index, request in enumerate(requests):
        if check_cancel:check_cancel()
        path=Path(request['path']); dataset=request['dataset']
        with h5py.File(path,'r') as handle:
            current=handle
            for component in dataset.strip('/').split('/'):
                if not isinstance(current,h5py.Group):raise ValueError('Invalid HDF5 group traversal.')
                link=current.get(component,getlink=True)
                if not isinstance(link,h5py.HardLink):
                    raise ValueError('HDF5 datasets must use local hard links only.')
                current=current[component]
            data=current
            if not isinstance(data,h5py.Dataset) or data.is_virtual or data.external:
                raise ValueError('External, virtual, or non-dataset HDF5 objects are unsupported.')
            expected=tuple(request['shape'])
            if data.shape != expected or data.dtype.kind not in 'fiu' or data.dtype.itemsize>8:
                raise ValueError('HDF5 dataset shape or numeric type disagrees with its XMF declaration.')
            if request['role']=='cells' and data.dtype.kind not in 'iu':
                raise ValueError('Triangle indices must be integers.')
            units=data.attrs.get('units',data.attrs.get('unit',None))
            if isinstance(units,bytes):units=units.decode('utf-8')
            if units is not None and not isinstance(units,str):
                raise ValueError('HDF5 units must be an explicit string.')
            name=f'dataset-{index:05d}.npy'; destination=output_dir/name
            if destination.exists():raise ValueError('Extraction refuses to overwrite an existing dataset.')
            array=np.lib.format.open_memmap(destination,mode='w+',dtype=data.dtype,shape=data.shape)
            for start in range(0,len(data),65536):
                if check_cancel:check_cancel()
                array[start:start+65536]=data[start:start+65536]
            array.flush(); del array
            result.append(dict(filename=name,shape=list(data.shape),dtype=str(data.dtype),units=units))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('request');parser.add_argument('output')
    args=parser.parse_args()
    request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    def check_cancel():
        if (Path(args.output)/'cancel').exists():raise InterruptedError('HDF5 extraction cancelled.')
    rows=extract(request,Path(args.output),check_cancel=check_cancel)
    (Path(args.output)/'extracted.json').write_text(json.dumps(rows),encoding='utf-8')
