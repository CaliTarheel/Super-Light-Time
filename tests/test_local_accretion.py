"""Local contact clocks select coherent material, not a whole moving plate."""
from copy import deepcopy
from types import SimpleNamespace, MethodType
import unittest

import numpy as np

from crust_transport import footprints
import local_accretion as local


def world(width=32, height=16, source=None, target=None):
    n = width*height
    lat = np.pi/2-(np.arange(height)+.5)*np.pi/height
    lon = (np.arange(width)+.5)*2*np.pi/width-np.pi
    longitude, latitude = np.meshgrid(lon, lat)
    xyz = np.column_stack((np.cos(latitude.ravel())*np.cos(longitude.ravel()),
                           np.cos(latitude.ravel())*np.sin(longitude.ravel()), np.sin(latitude.ravel())))
    rows = np.pi/2-np.arange(height+1)*np.pi/height
    area = np.repeat(-np.diff(np.sin(rows))*2*np.pi/width*6371**2, width)
    if source is None:
        source = [y*width+x for y in (7,8) for x in (6,7,18,19)]
    if target is None:
        target = [y*width+x for y in range(6,11) for x in range(8,13)]
        target += [y*width+x for y in range(6,11) for x in range(20,25)]
    plate, crust = np.full(n,2,np.int16), np.zeros(n,np.uint8)
    plate[source], plate[target] = 0,1
    crust[source], crust[target] = 1,1
    cells = np.r_[source,target]
    parcel_cells = np.repeat(cells,4)
    pos = xyz[parcel_cells].copy()
    east, extent = footprints(pos,width,height)
    grid=np.arange(n).reshape(height,width)
    north=np.vstack((np.roll(grid[0],width//2),grid[:-1])).ravel()
    south=np.vstack((grid[1:],np.roll(grid[-1],width//2))).ravel()
    def indices(points):
        lo=np.arctan2(points[:,1],points[:,0]); la=np.arcsin(np.clip(points[:,2],-1,1))
        return np.clip(((np.pi/2-la)*height/np.pi).astype(int),0,height-1)*width+(((lo+np.pi)*width/(2*np.pi)).astype(int)%width)
    s=SimpleNamespace(w=width,h=height,n=n,xyz=xyz,cell_area=area,plate=plate,crust=crust,
        pos=pos,mass=area[parcel_cells]/4,parcel_plate=plate[parcel_cells].copy(),
        parcel_patch=np.repeat(1000+cells,4).astype(np.int64),parcel_craton=np.full(len(pos),-1,np.int32),
        kind=np.ones(len(pos),np.uint8),parcel_east=east,parcel_extent=extent,
        trace_patch=(1000+cells).astype(np.int64),trace_plate=plate[cells].copy(),trace_xyz=xyz[cells].copy(),
        trace_kind=np.ones(len(cells),np.uint8),trace_origin_kind=np.ones(len(cells),np.uint8),
        trace_relief_m=np.full(len(cells),500.),trace_adjustment_m=np.zeros(len(cells)),trace_suture=np.full(len(cells),.2),
        relief=np.full(len(pos),500.),suture=np.full(len(pos),.2),active=np.ones(3,bool),plate_uid=np.array([41,52,63]),
        omega=np.array([[0,0,.006],[0,0,-.006],[0,0,0.]]),mantle=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]),
        born=np.zeros(3),t=100.,east=np.roll(grid,-1,axis=1).ravel(),west=np.roll(grid,1,axis=1).ravel(),
        north=north,south=south,_indices=indices,
        _sample_coordinates=lambda points:((indices(points),np.ones(len(points))),),
        ridge_episodes=[dict(id=1,center=[1,0,0],overriding_plate_uid=41)],
        backarc_basins=[],names=['Incoming','Receiver','Ocean'],process_totals={'accreted_km2':0.},events=[],
        rift_id=np.arange(len(pos)),rift_extension_m=np.arange(len(pos),dtype=float),
        rift_tangent=pos.copy(),inversion_uplift_m=np.arange(len(pos),dtype=float)/2)
    s.support=np.zeros((3,n));s.support[s.plate,np.arange(n)]=1
    s.structure={'thickness':np.arange(len(pos),dtype=float)}
    s.trace_structure={'thickness':np.arange(len(cells),dtype=float)}
    def record(self,typ,description,key=None,*,plates=(),xyz=None,details=None):
        self.events.append(dict(type=typ,description=description,key=key,plates=plates,details=deepcopy(details or {})))
    s._record=MethodType(record,s)
    contact(s,[(7*width+7,7*width+8),(8*width+7,8*width+8)])
    return s


def contact(s, pairs):
    s.ba=np.array([a for a,b in pairs],int);s.bb=np.array([b for a,b in pairs],int)
    s.bp=s.plate[s.ba];s.bq=s.plate[s.bb]
    mid=s.xyz[s.ba]+s.xyz[s.bb];mid/=np.linalg.norm(mid,axis=1)[:,None]
    normal=s.xyz[s.bb]-s.xyz[s.ba]
    normal-=mid*np.sum(normal*mid,axis=1)[:,None];normal/=np.linalg.norm(normal,axis=1)[:,None]
    s.bmid,s.bn=mid,normal;s.bcode=np.full(len(pairs),4,np.uint8);s.bl=np.full(len(pairs),5000.)


def mature(s,steps=12):
    result=[]
    for _ in range(steps):
        s.t+=2
        result=local.plan_accretions(s,2.)
        if result:
            return result[0]
    raise AssertionError('Prescribed sustained contact did not mature')


class LocalAccretionTests(unittest.TestCase):
    def test_only_contacting_terrane_and_matching_whole_patches_are_selected(self):
        s=world();before={k:v.copy() for k,v in vars(s).items() if isinstance(v,np.ndarray)}
        episodes=deepcopy((s.ridge_episodes,s.backarc_basins))
        plan=mature(s)
        expected=np.flatnonzero((s.parcel_plate==0)&(s._indices(s.pos)%s.w<10))
        np.testing.assert_array_equal(plan['parcel_indices'],expected)
        self.assertEqual((plan['source'],plan['target']),(0,1))
        self.assertEqual(plan['source_uid'],41)
        self.assertAlmostEqual(plan['area_km2'],float(s.mass[expected].sum()))
        self.assertTrue(np.all(np.isin(s.trace_patch[plan['trace_indices']],plan['patch_ids'])))
        self.assertEqual(len(plan['parcel_indices']),4*len(plan['patch_ids']))
        self.assertFalse(np.any(s._indices(s.pos[plan['parcel_indices']])%s.w>=18))
        for key,value in before.items():np.testing.assert_array_equal(getattr(s,key),value,err_msg=key)
        self.assertEqual((s.ridge_episodes,s.backarc_basins),episodes)

    def test_remote_contact_cannot_inherit_another_islands_loading(self):
        s=world()
        for _ in range(4):s.t+=2;self.assertEqual(local.plan_accretions(s,2.),[])
        old=s.local_accretion_contacts[0]['id']
        contact(s,[(7*s.w+19,7*s.w+20),(8*s.w+19,8*s.w+20)])
        s.t+=2;self.assertEqual(local.plan_accretions(s,2.),[])
        records=[r for r in s.local_accretion_contacts if r['last_seen_myr']==s.t]
        self.assertEqual(len(records),1);self.assertNotEqual(records[0]['id'],old)
        self.assertLess(records[0]['loading'],local.THRESHOLD)
        plan=mature(s)
        self.assertTrue(np.all(s._indices(s.pos[plan['parcel_indices']])%s.w>=18))

    def test_separate_margins_of_same_connected_terrane_have_local_clocks(self):
        # A solid source ribbon has two distant contacts to one solid receiver.
        w=32
        source=[7*w+x for x in range(4,25)]
        target=[8*w+x for x in range(4,25)]
        s=world(source=source,target=target)
        # Convergence normal is meridional in this fixture.
        s.omega[:2]=[[.006,0,0],[-.006,0,0]]
        contact(s,[(7*w+5,8*w+5)])
        if np.sum(np.cross(s.omega[1]-s.omega[0],s.bmid[0])*s.bn[0])>0:s.omega[:2]*=-1
        for _ in range(4):s.t+=2;local.plan_accretions(s,2.)
        old=[r['id'] for r in s.local_accretion_contacts]
        contact(s,[(7*w+23,8*w+23)])
        if np.sum(np.cross(s.omega[1]-s.omega[0],s.bmid[0])*s.bn[0])>0:s.omega[:2]*=-1
        s.t+=2;self.assertEqual(local.plan_accretions(s,2.),[])
        self.assertTrue(any(r['id'] not in old for r in s.local_accretion_contacts))

    def test_protected_craton_is_never_cut_by_local_selection(self):
        s=world()
        # A protected material ID remains one unit even when sampling conceals
        # part of its connection. This is not a freehand distance-radius cut.
        s.parcel_craton[s.parcel_plate==0]=9
        plan=mature(s)
        np.testing.assert_array_equal(plan['parcel_indices'],np.flatnonzero(s.parcel_plate==0))
        self.assertTrue(np.all(np.isin(np.flatnonzero(s.parcel_craton==9),plan['parcel_indices'])))

    def test_unresolved_arc_centres_do_not_bridge_two_continental_islands(self):
        s=world()
        cells=np.array([7*s.w+x for x in range(8,18)])
        extra=s.xyz[cells];east,extent=footprints(extra,s.w,s.h)
        s.pos=np.concatenate((s.pos,extra));s.mass=np.r_[s.mass,s.cell_area[cells]*.01]
        s.parcel_plate=np.r_[s.parcel_plate,np.zeros(len(cells),np.int16)]
        s.parcel_patch=np.r_[s.parcel_patch,5000+cells]
        s.parcel_craton=np.r_[s.parcel_craton,np.full(len(cells),-1)]
        s.kind=np.r_[s.kind,np.full(len(cells),3,np.uint8)]
        s.parcel_east=np.concatenate((s.parcel_east,east));s.parcel_extent=np.concatenate((s.parcel_extent,extent))
        plan=mature(s)
        self.assertFalse(np.any(s._indices(s.pos[plan['parcel_indices']])%s.w>=18))
        self.assertFalse(np.any(s.kind[plan['parcel_indices']]==3))

    def test_closing_motion_is_required_and_a_missing_contact_decays(self):
        s=world();s.t+=2;local.plan_accretions(s,2.)
        before=s.local_accretion_contacts[0]['loading']
        s.omega[:]=[0,0,.002]
        s.t+=2;self.assertEqual(local.plan_accretions(s,2.),[])
        self.assertLess(s.local_accretion_contacts[0]['loading'],before)
        self.assertEqual(len(s.local_accretion_contacts),1)
        s.omega[0,2]=-.006;s.omega[1,2]=.006
        s.t+=2;self.assertEqual(local.plan_accretions(s,2.),[])

    def test_coalescence_remaps_actual_contact_anchor_without_resetting_memory(self):
        s=world()
        for _ in range(4):s.t+=2;local.plan_accretions(s,2.)
        record=s.local_accretion_contacts[0];identity=record['id'];old=record['p_patch'];new=90000
        s.parcel_patch[s.parcel_patch==old]=new;s.trace_patch[s.trace_patch==old]=new
        local.remap_contact_patches(s,np.array([old]),np.array([new]))
        s.t+=2;local.plan_accretions(s,2.)
        self.assertEqual(s.local_accretion_contacts[0]['id'],identity)
        self.assertEqual(s.local_accretion_contacts[0]['p_patch'],new)
        self.assertEqual(len(s.local_accretion_contacts),1)
        self.assertGreater(s.local_accretion_contacts[0]['loading'],4.)

    def test_completed_contact_and_reused_plate_slot_cannot_reuse_a_debt(self):
        s=world();plan=mature(s);identity=plan['contact_id']
        local.complete_contact(s,identity)
        self.assertTrue(s.local_accretion_contacts[0]['completed'])
        s.plate_uid[0]=999;s.t+=2
        self.assertEqual(local.plan_accretions(s,2.),[])
        self.assertNotEqual(s.local_accretion_contacts[-1]['id'],identity)
        self.assertEqual(s.local_accretion_contacts[-1]['p_uid'],999)

    def test_seam_and_same_pole_material_components_remain_connected(self):
        for cells in ([7*32,7*32+31,8*32,8*32+31],
                      [0,1,16,17,32,33,48,49]):
            with self.subTest(cells=cells):
                s=world(source=cells,target=[8*32+10,8*32+11])
                geometry=local._material_components(s,0)
                self.assertEqual(len(geometry['components']),1)
                self.assertEqual(len(next(iter(geometry['components'].values()))['parcel_indices']),4*len(cells))

    def test_contact_state_roundtrip_preserves_the_next_plan(self):
        s=world()
        for _ in range(4):s.t+=2;local.plan_accretions(s,2.)
        other=deepcopy(s)
        a,b=mature(s),mature(other)
        self.assertEqual(a['contact_id'],b['contact_id']);self.assertEqual(a['loading'],b['loading'])
        np.testing.assert_array_equal(a['parcel_indices'],b['parcel_indices'])

    def test_application_keeps_remote_arcs_traces_ocean_and_structure_exact(self):
        s=world();s.kind[s.parcel_plate==0]=3;s.trace_kind[s.trace_plate==0]=3
        s.trace_origin_kind[s.trace_plate==0]=3;s.crust[s.plate==0]=3
        plan=mature(s)
        remote=(s.parcel_plate==0)&~np.isin(np.arange(len(s.mass)),plan['parcel_indices'])
        remote_traces=(s.trace_plate==0)&~np.isin(np.arange(len(s.trace_plate)),plan['trace_indices'])
        before={k:v.copy() for k,v in vars(s).items() if isinstance(v,np.ndarray)}
        structures=deepcopy((s.structure,s.trace_structure));mass=float(s.mass.sum())
        source_mass=plan['area_km2'];target_mass=float(s.mass[s.parcel_plate==1].sum())
        expected_omega=(s.omega[0]*source_mass+s.omega[1]*target_mass)/(source_mass+target_mass)
        expected_mantle=(s.mantle[0]*source_mass+s.mantle[1]*target_mass)/(source_mass+target_mass)
        old_height=220+s.relief[plan['parcel_indices']]-100
        old_trace_height=220+s.trace_relief_m[plan['trace_indices']]-100
        self.assertTrue(local.apply_accretion(s,plan))
        for name in ('pos','mass','parcel_patch','parcel_craton','parcel_east','parcel_extent','rift_id','rift_extension_m','rift_tangent','inversion_uplift_m','trace_xyz','trace_patch','trace_origin_kind'):
            np.testing.assert_array_equal(getattr(s,name),before[name],err_msg=name)
        for name in ('parcel_plate','kind','relief','suture'):
            np.testing.assert_array_equal(getattr(s,name)[remote],before[name][remote],err_msg=name)
        for name in ('trace_plate','trace_kind','trace_relief_m','trace_adjustment_m','trace_suture'):
            np.testing.assert_array_equal(getattr(s,name)[remote_traces],before[name][remote_traces],err_msg=name)
        np.testing.assert_allclose(s.omega[1],expected_omega);np.testing.assert_allclose(s.mantle[1],expected_mantle)
        np.testing.assert_array_equal(s.omega[0],before['omega'][0]);np.testing.assert_array_equal(s.mantle[0],before['mantle'][0])
        np.testing.assert_allclose(220+s.relief[plan['parcel_indices']],old_height)
        np.testing.assert_allclose(220+s.trace_relief_m[plan['trace_indices']],old_trace_height)
        np.testing.assert_array_equal(s.trace_adjustment_m[plan['trace_indices']],-100.)
        unaffected=np.ones(s.n,bool);unaffected[plan['surface_cells']]=False
        np.testing.assert_array_equal(s.support[:,unaffected],before['support'][:,unaffected])
        np.testing.assert_array_equal(s.plate[unaffected],before['plate'][unaffected])
        self.assertEqual(float(s.mass.sum()),mass);self.assertEqual(s.process_totals['accreted_km2'],source_mass)
        self.assertTrue(s.active[0]);self.assertEqual(s.plate_uid[0],41)
        for actual,old in zip((s.structure,s.trace_structure),structures):
            for key in old:np.testing.assert_array_equal(actual[key],old[key])
        self.assertEqual(s.events[-1]['type'],'accretion');self.assertEqual(s.events[-1]['details']['contact_id'],plan['contact_id'])
        self.assertFalse(local.apply_accretion(s,plan),'one plan cannot transfer twice')

    def test_only_local_slab_window_and_backarc_hosts_follow_transferred_crust(self):
        s=world();local_center=s.xyz[7*s.w+7].tolist();remote=s.xyz[7*s.w+19].tolist()
        s.ridge_episodes=[dict(id=1,center=local_center,overriding_plate_uid=41),dict(id=2,center=remote,overriding_plate_uid=41)]
        def basin(identity,center):return dict(id=identity,center=center,parent_plate_uid=52,arc_plate_uid=41,
            downgoing_plate_uid=63,phase='spreading',opening_km=100.,loading_km=50.,extension_rate_km_myr=2.)
        s.backarc_basins=[basin(1,local_center),basin(2,remote)]
        old=deepcopy(s.backarc_basins[1]);plan=mature(s)
        self.assertTrue(local.apply_accretion(s,plan))
        self.assertEqual([r['overriding_plate_uid'] for r in s.ridge_episodes],[52,41])
        self.assertEqual(s.backarc_basins[0]['phase'],'closed')
        self.assertEqual(s.backarc_basins[0]['arc_plate_uid'],52)
        self.assertEqual(s.backarc_basins[0]['opening_km'],100.)
        self.assertEqual(s.backarc_basins[1],old)
        event=[e for e in s.events if e['type']=='backarc_closed'][0]
        self.assertEqual(event['details']['backarc_id'],1)

    def test_stale_plan_fails_before_any_partial_transfer(self):
        s=world();plan=mature(s);s.plate_uid[0]=777
        before={k:v.copy() for k,v in vars(s).items() if isinstance(v,np.ndarray)}
        self.assertFalse(local.apply_accretion(s,plan))
        for key,value in before.items():np.testing.assert_array_equal(getattr(s,key),value)
        self.assertEqual(s.events,[]);self.assertEqual(s.process_totals['accreted_km2'],0.)


if __name__=='__main__':unittest.main()
