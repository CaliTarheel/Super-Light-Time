"""Validate the exported native mesh/forcing contract without installing goSPL."""
import json
from pathlib import Path
import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_package(root,check=None,progress=None):
    def checkpoint():
        if check:check()
    checkpoint()
    root=Path(root)
    config=json.loads((root/'input.yml').read_text(encoding='utf-8'))
    mesh=config['domain']['npdata']
    with np.load(root/(mesh[0]+'.npz'),allow_pickle=False) as data:
        v,c,z=(data[key] for key in mesh[1:])
    require(v.ndim==2 and v.shape[1]==3 and z.shape==(len(v),),'Invalid vertex/elevation shapes.')
    require(c.ndim==2 and c.shape[1]==3 and np.issubdtype(c.dtype,np.integer),'Invalid triangle indices.')
    require(np.isfinite(z).all(),'Non-finite initial elevation.')
    require(c.min()>=0 and c.max()<len(v),'Triangle index outside mesh.')
    for offset in range(0,len(v),65536):
        checkpoint();part=v[offset:offset+65536]
        require(np.isfinite(part).all(),'Non-finite vertex coordinates.')
        require(np.allclose(np.linalg.norm(part,axis=1),config['domain']['radius'],rtol=0,atol=1e-5),'Vertices do not lie on the declared sphere.')
    # Closed triangular manifold and spherical Euler characteristic.
    keys=[]
    for a,b in ((0,1),(1,2),(2,0)):
        checkpoint()
        lo=np.minimum(c[:,a],c[:,b]);hi=np.maximum(c[:,a],c[:,b])
        keys.append(lo.astype(np.int64)*len(v)+hi)
    edges,count=np.unique(np.concatenate(keys),return_counts=True)
    require(np.all(count==2) and len(v)-len(edges)+len(c)==2,'Mesh is not a closed spherical manifold.')
    degree=np.bincount(np.r_[edges//len(v),edges%len(v)],minlength=len(v))
    require(degree.min()>=3 and degree.max()<=12,'Mesh degree exceeds the goSPL neighbor contract.')
    del keys,edges,count,degree,lo,hi
    for offset in range(0,len(c),65536):
        checkpoint()
        a,b,d=np.moveaxis(v[c[offset:offset+65536]],1,0)
        require(np.all(np.einsum('ij,ij->i',np.cross(b-a,d-a),a)>0),'Triangles are degenerate or point inward.')
    now=config['time']['start'];dt=config['time']['dt'];n=len(v)
    require(config['domain']['advect']=='interp' and config['output']['makedir'] is True,'Expected interp advection and non-overwriting output mode.')
    require(dt>=1 and dt==int(dt),'Time step must use positive whole years.')
    for index,entry in enumerate(config['tectonics']):
        checkpoint()
        require(entry['start']==now and entry['end']>now and (entry['end']-now)%dt==0,'Tectonic intervals are discontinuous or do not align with the time step.')
        require(entry['hdisp'][0]==entry['upsub'][0],'This package validator expects horizontal and vertical fields in the same interval archive.')
        with np.load(root/(entry['hdisp'][0]+'.npz'),allow_pickle=False) as data:
            h=data[entry['hdisp'][1]];u=data[entry['upsub'][1]]
            require(h.shape==(n,3) and u.shape==(n,),'Forcing does not match mesh vertex count.')
            require(np.isfinite(h).all() and np.isfinite(u).all(),'Non-finite forcing values.')
            require(np.allclose(u,data['geometric_rate']+data['relaxation_correction'],rtol=2e-6,atol=2e-8),'Vertical forcing components do not add up.')
            for offset in range(0,n,65536):
                checkpoint();section=slice(offset,offset+65536)
                moved=v[section]+h[section]*(entry['end']-entry['start'])
                require(np.max(np.abs(np.linalg.norm(moved,axis=1)-config['domain']['radius']))<2.0,'Chord motion does not end on the sphere.')
        now=entry['end']
        if progress:progress((index+1)/len(config['tectonics']))
    require(now==config['time']['end'],'Forcing does not cover the configured duration.')
    return dict(passed=True,nodes=n,triangles=len(c),intervals=len(config['tectonics']),
                time_start_year=config['time']['start'],time_end_year=now,
                mesh_closed=True,units_checked=True,solver_run=False)


if __name__=='__main__':
    print(json.dumps(validate_package(Path(__file__).resolve().parent),indent=2))
