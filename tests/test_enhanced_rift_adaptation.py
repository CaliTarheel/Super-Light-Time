"""Newborn columns must not drop inherited weakness during immediate remeshing."""
import unittest
import numpy as np
import enhanced_rifting as enhanced
import native_material_adaptivity as adapt
from tests.test_native_material_adaptivity import world


class EnhancedRiftAdaptationTests(unittest.TestCase):
    def test_newborn_padding_precedes_real_refinement(self):
        s=world();s.config['enhanced_rifting']=enhanced.normalize({'enabled':True})
        enhanced.initialize(s);n=len(s.mass)
        # The newest appended material has no inherited initial weakness and
        # has not yet contributed a realized strain. Other native fields are
        # already aligned, exactly as after an ordinary arc birth transaction.
        for name in enhanced.ARRAY_FIELDS:
            target=name.replace('material_','parcel_',1)
            setattr(s,target,np.linspace(.01,.2,n-1))
        expected={name:float(s.mass@np.r_[getattr(s,name.replace('material_','parcel_',1)),0.]) for name in enhanced.ARRAY_FIELDS}
        self.assertTrue(adapt.adapt(s,2.));self.assertGreater(len(s.mass),n)
        for name in enhanced.ARRAY_FIELDS:
            value=getattr(s,name.replace('material_','parcel_',1))
            self.assertEqual(value.shape,s.mass.shape)
            self.assertAlmostEqual(float(s.mass@value),expected[name],delta=max(1e-7,expected[name]*2e-13))

    def test_schema_requires_strict_version_and_bounded_weakness(self):
        s=world();s.config['enhanced_rifting']=enhanced.normalize({'enabled':True});enhanced.initialize(s)
        frame=dict(material_faces=s.material_surface['faces'],**enhanced.snapshot_fields(s));enhanced.validate_frame(frame)
        frame['enhanced_rifting_version']=True
        with self.assertRaisesRegex(ValueError,'version'):enhanced.validate_frame(frame)
        frame['enhanced_rifting_version']=0
        with self.assertRaisesRegex(ValueError,'schema'):enhanced.validate_frame(frame)
        frame['enhanced_rifting_version']=1;frame['material_rift_seed_weakness'][0]=.9
        with self.assertRaisesRegex(ValueError,'amplitude'):enhanced.validate_frame(frame)


if __name__=='__main__':unittest.main()
